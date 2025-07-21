/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

using StreamingShuffler = BaseStreamingFixture;

TEST_F(StreamingShuffler, Basic) {
    constexpr unsigned int num_partitions = 10;
    constexpr unsigned int num_rows = 1000;
    constexpr unsigned int num_chunks = 5;
    constexpr unsigned int chunk_size = num_rows / num_chunks;
    constexpr std::int64_t seed = 42;
    constexpr cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
    constexpr OpID op_id = 0;

    // Create the full input table and slice it into chunks.
    cudf::table full_input_table = random_table_with_index(seed, num_rows, 0, 10);
    std::vector<std::unique_ptr<TableChunk>> full_input_table_chunks;
    for (unsigned int i = 0; i < num_chunks; ++i) {
        full_input_table_chunks.emplace_back(
            std::make_unique<TableChunk>(
                i,
                std::make_unique<cudf::table>(
                    cudf::slice(
                        full_input_table,
                        {static_cast<cudf::size_type>(i * chunk_size),
                         static_cast<cudf::size_type>((i + 1) * chunk_size)},
                        ctx->stream()
                    )
                        .at(0),
                    ctx->stream(),
                    ctx->br()->device_mr()
                )
            )
        );
    }

    // Create and run the streaming pipeline.
    std::vector<std::unique_ptr<TableChunk>> output_chunks;
    {
        std::vector<Node> nodes;
        auto ch1 = make_shared_channel<TableChunk>();
        nodes.push_back(
            node::push_chunks_to_channel<TableChunk>(
                ctx, ch1, std::move(full_input_table_chunks)
            )
        );

        auto ch2 = make_shared_channel<PartitionMapChunk>();
        nodes.push_back(
            node::partition_and_pack(
                ctx, ch1, ch2, {1}, num_partitions, hash_function, seed
            )
        );

        auto ch3 = make_shared_channel<PartitionVectorChunk>();
        nodes.push_back(node::shuffler(ctx, ch2, ch3, op_id, num_partitions));

        auto ch4 = make_shared_channel<TableChunk>();
        nodes.push_back(node::unpack_and_concat(ctx, ch3, ch4));

        nodes.push_back(node::pull_chunks_from_channel(ctx, ch4, output_chunks));

        run_streaming_pipeline(std::move(nodes));
    }

    // Concat all output chunks to a single table.
    std::vector<cudf::table_view> output_chunks_as_views;
    for (auto const& chunk : output_chunks) {
        output_chunks_as_views.push_back(chunk->table_view());
    }
    auto result_table = cudf::concatenate(output_chunks_as_views);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
        sort_table(result_table->view()), sort_table(full_input_table.view())
    );
}
