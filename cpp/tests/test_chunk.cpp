/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

#include "utils.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::shuffler;
using namespace rapidsmpf::shuffler::detail;

class ChunkTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
        stream = cudf::get_default_stream();
    }

    std::unique_ptr<BufferResource> br;
    rmm::cuda_stream_view stream;
};

TEST_F(ChunkTest, FromFinishedPartition) {
    ChunkID chunk_id = 123;
    PartID part_id = 456;
    size_t expected_num_chunks = 789;

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.n_messages(), 1);
        EXPECT_EQ(chunk.part_id(0), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(0), expected_num_chunks);
        EXPECT_TRUE(chunk.is_control_message(0));
        EXPECT_EQ(chunk.metadata_size(0), 0);
        EXPECT_EQ(chunk.data_size(0), 0);
    };

    auto chunk = Chunk::from_finished_partition(chunk_id, part_id, expected_num_chunks);
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, true);
    test_chunk(chunk2);

    auto chunk3 = chunk2.get_data(chunk_id, 0, br.get());
    test_chunk(chunk3);
}

TEST_F(ChunkTest, FromPackedData) {
    ChunkID chunk_id = 123;
    PartID part_id = 456;

    // Create test metadata
    auto metadata =
        std::make_unique<std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3, 4});

    // Create test GPU data
    auto data = std::make_unique<rmm::device_buffer>(4, cudf::get_default_stream());
    std::vector<uint8_t> host_data{5, 6, 7, 8};
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(data->data(), host_data.data(), 4, cudaMemcpyHostToDevice)
    );

    PackedData packed_data{
        std::make_unique<std::vector<uint8_t>>(*metadata),
        br->move(std::move(data), stream)
    };

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.n_messages(), 1);
        EXPECT_EQ(chunk.part_id(0), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(0), 0);
        EXPECT_FALSE(chunk.is_control_message(0));
        EXPECT_EQ(chunk.metadata_size(0), 4);
        EXPECT_EQ(chunk.data_size(0), 4);
    };

    // no need of an event because cuda buffer copy is synchronous
    auto chunk = Chunk::from_packed_data(chunk_id, part_id, std::move(packed_data));
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, true);
    chunk2.set_data_buffer(chunk.release_data_buffer());
    test_chunk(chunk2);

    auto chunk3 = chunk2.get_data(chunk_id, 0, br.get());
    test_chunk(chunk3);
}
