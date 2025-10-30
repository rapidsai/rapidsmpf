/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rapidsmpf/allgather/allgather.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "base_streaming_fixture.hpp"


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

class StreamingReadParquet : public BaseStreamingFixture {
  protected:
    void SetUp() override {
        BaseStreamingFixture::SetUp();
        constexpr int nfiles = 10;
        constexpr int nrows = 10;

        temp_dir = std::filesystem::temp_directory_path() / "rapidsmpf_read_parquet_test";

        for (int i = 0; i < nfiles; ++i) {
            std::ostringstream filename_stream;
            filename_stream << std::setw(3) << std::setfill(' ') << i << ".pq";
            std::filesystem::path filepath = temp_dir / filename_stream.str();
            source_files.push_back(filepath.string());
        }

        if (GlobalEnvironment->comm_->rank() == 0) {
            std::filesystem::create_directories(temp_dir);

            int start = 0;
            for (auto& file : source_files) {
                auto values = std::ranges::iota_view(start, start + nrows);
                cudf::test::fixed_width_column_wrapper<int32_t> col(
                    values.begin(), values.end()
                );

                std::vector<std::unique_ptr<cudf::column>> columns;
                columns.push_back(col.release());
                auto table = std::make_unique<cudf::table>(std::move(columns));

                cudf::io::sink_info sink{file};
                auto options =
                    cudf::io::parquet_writer_options::builder(sink, table->view())
                        .build();
                cudf::io::write_parquet(options);
                start += nrows + nrows / 2;
            }
        }

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();

        if (GlobalEnvironment->comm_->rank() == 0 && std::filesystem::exists(temp_dir)) {
            std::filesystem::remove_all(temp_dir);
        }

        BaseStreamingFixture::TearDown();
    }

    [[nodiscard]] cudf::io::source_info get_source_info() const {
        return cudf::io::source_info(source_files);
    }

    std::filesystem::path temp_dir;
    std::vector<std::string> source_files;
};

using ReadParquetParams = std::tuple<std::optional<int64_t>, std::optional<int64_t>>;

class StreamingReadParquetParams
    : public StreamingReadParquet,
      public ::testing::WithParamInterface<ReadParquetParams> {};

INSTANTIATE_TEST_SUITE_P(
    ReadParquetCombinations,
    StreamingReadParquetParams,
    ::testing::Combine(
        // skip_rows
        ::testing::Values(
            std::nullopt,
            std::optional<int64_t>{7},
            std::optional<int64_t>{19},
            std::optional<int64_t>{113}
        ),
        // num_rows
        ::testing::Values(
            std::nullopt,
            std::optional<int64_t>{0},
            std::optional<int64_t>{3},
            std::optional<int64_t>{31},
            std::optional<int64_t>{83}
        )
    ),
    [](const ::testing::TestParamInfo<ReadParquetParams>& info) {
        const auto& skip_rows = std::get<0>(info.param);
        const auto& num_rows = std::get<1>(info.param);
        std::string result = "skip_rows_";
        result += skip_rows.has_value() ? std::to_string(skip_rows.value()) : "none";
        result += "_num_rows_";
        result += num_rows.has_value() ? std::to_string(num_rows.value()) : "all";
        return result;
    }
);

TEST_P(StreamingReadParquetParams, ReadParquet) {
    auto [skip_rows, num_rows] = GetParam();
    auto source = get_source_info();

    auto options = cudf::io::parquet_reader_options::builder(source).build();
    if (skip_rows.has_value()) {
        options.set_skip_rows(skip_rows.value());
    }
    if (num_rows.has_value()) {
        options.set_num_rows(num_rows.value());
    }

    auto ch = std::make_shared<Channel>();
    std::vector<Node> nodes;

    nodes.push_back(node::read_parquet(ctx, ch, 4, options, 3));

    std::vector<Message> messages;
    nodes.push_back(node::pull_from_channel(ctx, ch, messages));

    if (GlobalEnvironment->comm_->nranks() > 1
        && (skip_rows.value_or(0) > 0 || num_rows.has_value()))
    {
        // We don't yet implement skip_rows/num_rows in multi-rank mode
        EXPECT_THROW(run_streaming_pipeline(std::move(nodes)), std::logic_error);
        return;
    }
    run_streaming_pipeline(std::move(nodes));

    allgather::AllGather allgather(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        /* op_id = */ 0,
        br.get()
    );

    for (auto& msg : messages) {
        auto chunk = msg.release<TableChunk>();
        auto seq = msg.sequence_number();
        auto [reservation, _] =
            br->reserve(MemoryType::DEVICE, chunk.make_available_cost(), true);
        chunk = chunk.make_available(reservation);
        auto packed_columns =
            cudf::pack(chunk.table_view(), chunk.stream(), br->device_mr());
        auto packed_data = PackedData{
            std::move(packed_columns.metadata),
            br->move(std::move(packed_columns.gpu_data), chunk.stream())
        };

        allgather.insert(seq, std::move(packed_data));
    }

    allgather.insert_finished();

    // May as well check on all ranks, so we also mildly exercise the allgather.
    auto gathered_packed_data =
        allgather.wait_and_extract(allgather::AllGather::Ordered::YES);
    auto result = unpack_and_concat(
        std::move(gathered_packed_data), br->stream_pool().get_stream(), br.get()
    );
    auto expected = cudf::io::read_parquet(options).tbl;

    EXPECT_EQ(result->num_rows(), expected->num_rows());
    EXPECT_EQ(result->num_columns(), expected->num_columns());
    EXPECT_EQ(result->num_columns(), 1);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected->view());
}
