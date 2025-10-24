/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/cudf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingTableChunk = BaseStreamingFixture;

TEST_F(StreamingTableChunk, FromTable) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    TableChunk chunk{seq, std::make_unique<cudf::table>(expect), stream};
    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, TableChunkOwner) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    // Static because the deleter function is a void(*)(void*) which precludes the use of
    // a lambda with captures.
    static std::size_t num_deletions{0};
    auto deleter = [](void* p) {
        num_deletions++;
        delete static_cast<int*>(p);
    };
    auto make_chunk = [&](TableChunk::ExclusiveView exclusive_view) {
        return TableChunk{
            seq,
            expect,
            expect.alloc_size(),
            stream,
            OwningWrapper(new int, deleter),
            exclusive_view
        };
    };
    auto check_chunk = [&](TableChunk const& chunk, bool is_spillable) {
        EXPECT_EQ(chunk.sequence_number(), seq);
        EXPECT_EQ(chunk.stream().value(), stream.value());
        EXPECT_TRUE(chunk.is_available());
        EXPECT_EQ(chunk.is_spillable(), is_spillable);
        EXPECT_EQ(chunk.make_available_cost(), 0);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
    };
    {
        auto chunk = make_chunk(TableChunk::ExclusiveView::NO);
        check_chunk(chunk, false);
        EXPECT_EQ(num_deletions, 0);
    }
    EXPECT_EQ(num_deletions, 1);
    {
        auto msg = Message(
            std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::NO))
        );
        EXPECT_EQ(num_deletions, 1);
    }
    EXPECT_EQ(num_deletions, 2);
    {
        auto msg = Message(
            std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::YES))
        );
        auto chunk = msg.release<TableChunk>();
        check_chunk(chunk, true);
        EXPECT_EQ(num_deletions, 2);
    }
    EXPECT_EQ(num_deletions, 3);
    {
        auto chunk = make_chunk(TableChunk::ExclusiveView::YES);
        check_chunk(chunk, true);
        chunk = chunk.spill_to_host(br.get());
        EXPECT_EQ(num_deletions, 4);
    }
    {
        auto chunk = make_chunk(TableChunk::ExclusiveView::NO);
        check_chunk(chunk, false);
        EXPECT_THROW(std::ignore = chunk.spill_to_host(br.get()), std::invalid_argument);
    }
}

TEST_F(StreamingTableChunk, FromPackedColumns) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed = cudf::pack(expect, stream);

    TableChunk chunk{
        seq, std::make_unique<cudf::packed_columns>(std::move(packed)), stream
    };

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, FromPackedDataOnDevice) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        br->move(std::move(packed_columns.gpu_data), stream)
    );
    TableChunk chunk{seq, std::move(packed_data)};

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);

    // Eventhough the table isn't available, it is still all in device memory
    // so the memory cost of making it available is zero.
    EXPECT_EQ(chunk.make_available_cost(), 0);
}

TEST_F(StreamingTableChunk, FromPackedDataOnHost) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);
    std::size_t const size = packed_columns.gpu_data->size();

    // Move the gpu_data to a Buffer (still device memory).
    auto gpu_data_on_device = br->move(std::move(packed_columns.gpu_data), stream);

    // Copy the GPU data to host memory.
    auto [res, _] = br->reserve(MemoryType::HOST, size, true);
    auto gpu_data_on_host = br->move(std::move(gpu_data_on_device), res);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata), std::move(gpu_data_on_host)
    );
    TableChunk chunk{seq, std::move(packed_data)};

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);
    EXPECT_EQ(chunk.make_available_cost(), size);
}

TEST_F(StreamingTableChunk, SpillUnspillRoundTrip) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    TableChunk chunk_on_device{seq, std::make_unique<cudf::table>(expect), stream};
    EXPECT_TRUE(chunk_on_device.is_available());
    EXPECT_TRUE(chunk_on_device.is_spillable());

    // Spill to host memory.
    TableChunk chunk_on_host = chunk_on_device.spill_to_host(br.get());
    EXPECT_FALSE(chunk_on_host.is_available());
    // We are allowed to spill an already spilled chunk.
    chunk_on_host = chunk_on_host.spill_to_host(br.get());
    EXPECT_FALSE(chunk_on_host.is_available());

    // Unspill back to device memory.
    auto [res, _] =
        br->reserve(MemoryType::DEVICE, chunk_on_host.make_available_cost(), true);
    chunk_on_device = chunk_on_host.make_available(res);

    EXPECT_EQ(chunk_on_device.sequence_number(), seq);
    EXPECT_EQ(chunk_on_device.stream().value(), stream.value());
    EXPECT_TRUE(chunk_on_device.is_available());
    EXPECT_EQ(chunk_on_device.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk_on_device.table_view(), expect);
}
