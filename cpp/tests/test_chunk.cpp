/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {
namespace test {

class ChunkBatchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
    }

    std::unique_ptr<BufferResource> br;
};

TEST_F(ChunkBatchTest, FromFinishedPartition) {
    ChunkID chunk_id = 123;
    PartID part_id = 456;
    size_t expected_num_chunks = 789;

    auto test_chunk = [&](ChunkBatch& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.n_messages(), 1);
        EXPECT_EQ(chunk.part_id(0), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(0), expected_num_chunks);
        EXPECT_TRUE(chunk.is_control_message(0));
        EXPECT_EQ(chunk.metadata_size(0), 0);
        EXPECT_EQ(chunk.data_size(0), 0);
    };

    auto chunk =
        ChunkBatch::from_finished_partition(chunk_id, part_id, expected_num_chunks);
    test_chunk(chunk);

    auto chunk2 = ChunkBatch::from_metadata_message(chunk.release_metadata_buffer());
    test_chunk(chunk2);

    auto chunk3 = chunk2.get_data(chunk_id, 0, cudf::get_default_stream());
    test_chunk(chunk3);
}

// TEST_F(ChunkBatchTest, FromPackedData) {
//     ChunkID chunk_id = 123;
//     PartID part_id = 456;

//     // Create test metadata
//     auto metadata = std::make_unique<std::vector<uint8_t>>(4);
//     metadata->at(0) = 1;
//     metadata->at(1) = 2;
//     metadata->at(2) = 3;
//     metadata->at(3) = 4;

//     // Create test GPU data
//     auto gpu_data = br->allocate(MemoryType::DEVICE, 4);
//     uint8_t host_data[4] = {5, 6, 7, 8};
//     CUDA_TRY(cudaMemcpy(gpu_data->data, host_data, 4, cudaMemcpyHostToDevice));

//     PackedData packed_data{std::move(metadata), std::move(gpu_data)};
//     auto chunk =
//         ChunkBatch::from_packed_data(chunk_id, part_id, std::move(packed_data),
//         br.get());

//     EXPECT_EQ(chunk.chunk_id(), chunk_id);
//     EXPECT_EQ(chunk.n_messages(), 1);
//     EXPECT_EQ(chunk.part_id(0), part_id);
//     EXPECT_EQ(chunk.expected_num_chunks(0), 0);
//     EXPECT_FALSE(chunk.is_control_message(0));
//     EXPECT_EQ(chunk.metadata_size(0), 4);
//     EXPECT_EQ(chunk.data_size(0), 4);
// }

// TEST_F(ChunkBatchTest, ValidateMetadataFormat) {
//     // Test valid metadata format
//     auto valid_chunk = ChunkBatch::from_finished_partition(123, 456, 789);
//     EXPECT_TRUE(ChunkBatch::validate_metadata_format(*valid_chunk.release_metadata_buffer(
//     )));

//     // Test invalid metadata format (too small)
//     std::vector<uint8_t> too_small(4);
//     EXPECT_FALSE(ChunkBatch::validate_metadata_format(too_small));

//     // Test invalid metadata format (zero messages)
//     std::vector<uint8_t> zero_messages(16);
//     *reinterpret_cast<ChunkID*>(zero_messages.data()) = 123;
//     *reinterpret_cast<size_t*>(zero_messages.data() + sizeof(ChunkID)) = 0;
//     EXPECT_FALSE(ChunkBatch::validate_metadata_format(zero_messages));
// }

// TEST_F(ChunkBatchTest, GetData) {
//     ChunkID chunk_id = 123;
//     PartID part_id = 456;

//     // Create test metadata
//     auto metadata = std::make_unique<std::vector<uint8_t>>(4);
//     metadata->at(0) = 1;
//     metadata->at(1) = 2;
//     metadata->at(2) = 3;
//     metadata->at(3) = 4;

//     // Create test GPU data
//     auto gpu_data = br->allocate(MemoryType::DEVICE, 4);
//     uint8_t host_data[4] = {5, 6, 7, 8};
//     CUDA_TRY(cudaMemcpy(gpu_data->data, host_data, 4, cudaMemcpyHostToDevice));

//     PackedData packed_data{std::move(metadata), std::move(gpu_data)};
//     auto chunk =
//         ChunkBatch::from_packed_data(chunk_id, part_id, std::move(packed_data),
//         br.get());

//     // Test getting data from a data message
//     auto new_chunk = chunk.get_data(789, 0, cudf::get_default_stream());
//     EXPECT_EQ(new_chunk.chunk_id(), 789);
//     EXPECT_EQ(new_chunk.n_messages(), 1);
//     EXPECT_EQ(new_chunk.part_id(0), part_id);
//     EXPECT_EQ(new_chunk.metadata_size(0), 4);
//     EXPECT_EQ(new_chunk.data_size(0), 4);

//     // Test getting data from a control message
//     auto control_chunk = ChunkBatch::from_finished_partition(123, 456, 789);
//     auto new_control_chunk = control_chunk.get_data(999, 0,
//     cudf::get_default_stream()); EXPECT_EQ(new_control_chunk.chunk_id(), 999);
//     EXPECT_EQ(new_control_chunk.n_messages(), 1);
//     EXPECT_EQ(new_control_chunk.part_id(0), 456);
//     EXPECT_EQ(new_control_chunk.expected_num_chunks(0), 789);
//     EXPECT_TRUE(new_control_chunk.is_control_message(0));
// }

// TEST_F(ChunkBatchTest, FromMetadataMessage) {
//     // Create a chunk and convert it to metadata message
//     auto original_chunk = ChunkBatch::from_finished_partition(123, 456, 789);
//     auto metadata_msg = original_chunk.release_metadata_buffer();

//     // Create new chunk from metadata message
//     auto new_chunk = ChunkBatch::from_metadata_message(std::move(metadata_msg));

//     EXPECT_EQ(new_chunk.chunk_id(), 123);
//     EXPECT_EQ(new_chunk.n_messages(), 1);
//     EXPECT_EQ(new_chunk.part_id(0), 456);
//     EXPECT_EQ(new_chunk.expected_num_chunks(0), 789);
//     EXPECT_TRUE(new_chunk.is_control_message(0));
// }

}  // namespace test
}  // namespace rapidsmpf::shuffler::detail