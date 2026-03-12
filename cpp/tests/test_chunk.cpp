/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <numeric>

#include <driver_types.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

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
    std::size_t expected_num_chunks = 789;

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.part_id(), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(), expected_num_chunks);
        EXPECT_TRUE(chunk.is_control_message());
        EXPECT_EQ(chunk.metadata_size(), 0);
        EXPECT_EQ(chunk.data_size(), 0);
    };

    auto chunk = Chunk::from_finished_partition(chunk_id, part_id, expected_num_chunks);
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, br.get(), true);
    test_chunk(chunk2);
}

class ChunkFromPackedDataTest : public ChunkTest,
                                public ::testing::WithParamInterface<std::size_t> {};

TEST_P(ChunkFromPackedDataTest, RoundTrip) {
    std::size_t const data_size = GetParam();
    ChunkID chunk_id = 123;
    PartID part_id = 456;

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        std::vector<std::uint8_t>{1, 2, 3, 4}
    );

    auto data =
        std::make_unique<rmm::device_buffer>(data_size, cudf::get_default_stream());
    if (data_size > 0) {
        std::vector<std::uint8_t> host_data(data_size);
        std::iota(host_data.begin(), host_data.end(), std::uint8_t{5});
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpy(data->data(), host_data.data(), data_size, cudaMemcpyDefault)
        );
    }

    PackedData packed_data{std::move(metadata), br->move(std::move(data), stream)};

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.part_id(), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(), 0);
        EXPECT_FALSE(chunk.is_control_message());
        EXPECT_EQ(chunk.metadata_size(), 4);
        EXPECT_EQ(chunk.data_size(), data_size);
        EXPECT_TRUE(chunk.is_data_buffer_set());
    };

    auto chunk = Chunk::from_packed_data(chunk_id, part_id, std::move(packed_data));
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, br.get(), true);
    test_chunk(chunk2);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkFromPackedData, ChunkFromPackedDataTest, ::testing::Values(0, 4)
);
