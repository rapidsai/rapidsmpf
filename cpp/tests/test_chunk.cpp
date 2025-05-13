/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda/std/span>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
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

namespace {

/// @brief Create a PackedData object from a host buffer
PackedData create_packed_data(
    cuda::std::span<uint8_t const> metadata,
    cuda::std::span<uint8_t const> data,
    rmm::cuda_stream_view stream
) {
    auto metadata_ptr =
        std::make_unique<std::vector<uint8_t>>(metadata.begin(), metadata.end());
    auto data_ptr = std::make_unique<rmm::device_buffer>(data.size(), stream);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(data_ptr->data(), data.data(), data.size(), cudaMemcpyHostToDevice)
    );
    return PackedData{std::move(metadata_ptr), std::move(data_ptr)};
}

}  // namespace

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

    auto chunk3 = chunk2.get_data(chunk_id, 0, stream);
    test_chunk(chunk3);

    EXPECT_THROW(chunk3.get_data(chunk_id, 1, stream), std::out_of_range);
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
        std::make_unique<std::vector<uint8_t>>(*metadata), std::move(data)
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
    auto chunk = Chunk::from_packed_data(
        chunk_id, part_id, std::move(packed_data), nullptr, stream, br.get()
    );
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, true);
    chunk2.set_data_buffer(chunk.release_data_buffer());
    test_chunk(chunk2);

    auto chunk3 = chunk2.get_data(chunk_id, 0, stream);
    test_chunk(chunk3);
}

TEST_F(ChunkTest, ChunkBuilderControlMessages) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(chunk_id, stream, br.get(), 3);  // Hint for 3 messages

    // Add three control messages
    builder.add_control_message(1, 10).add_control_message(2, 20).add_control_message(
        3, 30
    );

    auto chunk = builder.build();

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 3);

    // Verify each message
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(chunk.part_id(i), i + 1);
        EXPECT_EQ(chunk.expected_num_chunks(i), (i + 1) * 10);
        EXPECT_TRUE(chunk.is_control_message(i));
        EXPECT_EQ(chunk.metadata_size(i), 0);
        EXPECT_EQ(chunk.data_size(i), 0);
    }
}

TEST_F(ChunkTest, ChunkBuilderPackedData) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(chunk_id, stream, br.get(), 2);  // Hint for 2 messages

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data

    // Add the packed data messages using spans
    builder
        .add_packed_data(
            1, create_packed_data({metadata.data(), 3}, {data.data(), 3}, stream)
        )
        .add_packed_data(
            2, create_packed_data({metadata.data() + 3, 2}, {data.data() + 3, 2}, stream)
        );

    auto chunk = builder.build();

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 2);

    // Verify first message
    EXPECT_EQ(chunk.part_id(0), 1);
    EXPECT_EQ(chunk.expected_num_chunks(0), 0);
    EXPECT_FALSE(chunk.is_control_message(0));
    EXPECT_EQ(chunk.metadata_size(0), 3);
    EXPECT_EQ(chunk.data_size(0), 3);

    // Verify second message
    EXPECT_EQ(chunk.part_id(1), 2);
    EXPECT_EQ(chunk.expected_num_chunks(1), 0);
    EXPECT_FALSE(chunk.is_control_message(1));
    EXPECT_EQ(chunk.metadata_size(1), 2);
    EXPECT_EQ(chunk.data_size(1), 2);

    // Release and verify buffers
    auto released_metadata = chunk.release_metadata_buffer();
    auto released_data = chunk.release_data_buffer();

    // Verify metadata buffer
    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(released_metadata->size(), 5);  // Total size of both metadata chunks
    EXPECT_EQ(metadata, *released_metadata);

    // Verify data buffer
    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, 5);  // Total size of both data chunks
    std::vector<uint8_t> host_data(5);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(host_data.data(), released_data->data(), 5, cudaMemcpyDeviceToHost)
    );
    EXPECT_EQ(data, host_data);
}

TEST_F(ChunkTest, ChunkBuilderMixedMessages) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(chunk_id, stream, br.get(), 4);  // Hint for 4 messages

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data


    auto chunk =
        builder
            .add_control_message(1, 10)  // control message 1
            .add_packed_data(
                2, create_packed_data({metadata.data(), 3}, {data.data(), 3}, stream)
            )  // packed data 1
            .add_control_message(3, 40)  // control message 2
            .add_packed_data(
                4,
                create_packed_data({metadata.data() + 5, 0}, {data.data() + 5, 0}, stream)
            )  // empty packed data - non-null
            .add_packed_data(
                5,
                create_packed_data({metadata.data() + 3, 2}, {data.data() + 3, 2}, stream)
            )  // packed data 2
            .add_packed_data(6, PackedData{nullptr, nullptr})  // empty packed data - null
            .build();

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 6);

    // Verify control message 1
    EXPECT_EQ(chunk.part_id(0), 1);
    EXPECT_EQ(chunk.expected_num_chunks(0), 10);
    EXPECT_TRUE(chunk.is_control_message(0));
    EXPECT_EQ(chunk.metadata_size(0), 0);
    EXPECT_EQ(chunk.data_size(0), 0);

    // Verify first packed data message
    EXPECT_EQ(chunk.part_id(1), 2);
    EXPECT_EQ(chunk.expected_num_chunks(1), 0);
    EXPECT_FALSE(chunk.is_control_message(1));
    EXPECT_EQ(chunk.metadata_size(1), 3);
    EXPECT_EQ(chunk.data_size(1), 3);

    // Verify control message 2
    EXPECT_EQ(chunk.part_id(2), 3);
    EXPECT_EQ(chunk.expected_num_chunks(2), 40);
    EXPECT_TRUE(chunk.is_control_message(2));
    EXPECT_EQ(chunk.metadata_size(2), 0);
    EXPECT_EQ(chunk.data_size(2), 0);

    // Verify empty packed data message
    EXPECT_EQ(chunk.part_id(3), 4);
    EXPECT_EQ(chunk.expected_num_chunks(3), 0);
    EXPECT_FALSE(chunk.is_control_message(3));
    EXPECT_EQ(chunk.metadata_size(3), 0);
    EXPECT_EQ(chunk.data_size(3), 0);

    // Verify second packed data message
    EXPECT_EQ(chunk.part_id(4), 5);
    EXPECT_EQ(chunk.expected_num_chunks(4), 0);
    EXPECT_FALSE(chunk.is_control_message(4));
    EXPECT_EQ(chunk.metadata_size(4), 2);
    EXPECT_EQ(chunk.data_size(4), 2);

    // Verify empty packed data message with null metadata and data
    EXPECT_EQ(chunk.part_id(5), 6);
    EXPECT_EQ(chunk.expected_num_chunks(5), 0);
    EXPECT_FALSE(chunk.is_control_message(5));
    EXPECT_EQ(chunk.metadata_size(5), 0);
    EXPECT_EQ(chunk.data_size(5), 0);

    // Release and verify buffers
    auto released_metadata = chunk.release_metadata_buffer();
    auto released_data = chunk.release_data_buffer();

    // Verify metadata buffer
    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(metadata, *released_metadata);

    // Verify data buffer
    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, 5);  // Total size of data
    std::vector<uint8_t> host_data(5);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(host_data.data(), released_data->data(), 5, cudaMemcpyDeviceToHost)
    );
    EXPECT_EQ(data, host_data);
}

TEST_F(ChunkTest, ChunkWithHostBuffer) {
    // create a new buffer resource with only host memory available
    br = std::make_unique<BufferResource>(
        cudf::get_current_device_resource_ref(),
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{
            {MemoryType::DEVICE, []() { return 0; }}
        }
    );

    ChunkID chunk_id = 123;
    ChunkBuilder builder(chunk_id, stream, br.get(), 2);

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data

    auto chunk =
        builder.add_packed_data(1, create_packed_data(metadata, data, stream)).build();

    EXPECT_EQ(MemoryType::HOST, chunk.data_memory_type());
}
