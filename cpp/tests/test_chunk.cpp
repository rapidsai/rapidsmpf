/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>

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

    auto chunk3 = chunk2.get_data(chunk_id, 0, stream, br.get());
    test_chunk(chunk3);

    EXPECT_THROW(chunk3.get_data(chunk_id, 1, stream, br.get()), std::out_of_range);
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

    auto chunk3 = chunk2.get_data(chunk_id, 0, stream, br.get());
    test_chunk(chunk3);
}

TEST_F(ChunkTest, ChunkBuilderControlMessages) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(stream, br.get(), 3);  // Hint for 3 messages

    // No messages added to the builder -> should throw
    EXPECT_THROW(std::ignore = builder.build(0), std::runtime_error);

    // Add three control messages
    builder.add_control_message(1, 10).add_control_message(2, 20).add_control_message(
        3, 30
    );

    auto chunk = builder.build(chunk_id);

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 3);

    auto test_chunk = [&](size_t i, PartID part_id, size_t expected_num_chunks) {
        EXPECT_EQ(chunk.part_id(i), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(i), expected_num_chunks);
        EXPECT_TRUE(chunk.is_control_message(i));
        EXPECT_EQ(chunk.metadata_size(i), 0);
        EXPECT_EQ(chunk.data_size(i), 0);

        auto chunk_copy = chunk.get_data(chunk_id, i, stream, br.get());
        EXPECT_EQ(chunk_copy.part_id(0), part_id);
        EXPECT_EQ(chunk_copy.expected_num_chunks(0), expected_num_chunks);
        EXPECT_TRUE(chunk_copy.is_control_message(0));
        EXPECT_EQ(chunk_copy.metadata_size(0), 0);
        EXPECT_EQ(chunk_copy.data_size(0), 0);
    };

    test_chunk(0, 1, 10);
    test_chunk(1, 2, 20);
    test_chunk(2, 3, 30);

    // this is not the intended use of the release_metadata_buffer method, but this
    EXPECT_EQ(chunk.release_metadata_buffer()->size(), 0);
    EXPECT_EQ(chunk.release_data_buffer()->size, 0);

    // after building, the builder is empty
    EXPECT_THROW(std::ignore = builder.build(0), std::runtime_error);
}

TEST_F(ChunkTest, ChunkBuilderPackedData) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(stream, br.get(), 2);  // Hint for 2 messages

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

    auto chunk = builder.build(chunk_id);

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 2);

    auto test_chunk = [&](size_t i, PartID part_id, size_t size) {
        EXPECT_EQ(chunk.part_id(i), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(i), 0);
        EXPECT_FALSE(chunk.is_control_message(i));
        EXPECT_EQ(chunk.metadata_size(i), size);
        EXPECT_EQ(chunk.data_size(i), size);

        auto chunk_copy = chunk.get_data(chunk_id, i, stream, br.get());
        EXPECT_EQ(chunk_copy.part_id(0), part_id);
        EXPECT_EQ(chunk_copy.expected_num_chunks(0), 0);
        EXPECT_FALSE(chunk_copy.is_control_message(0));
        EXPECT_EQ(chunk_copy.metadata_size(0), size);
        EXPECT_EQ(chunk_copy.data_size(0), size);
    };

    test_chunk(0, 1, 3);
    test_chunk(1, 2, 2);

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

    // after building, the builder is empty
    EXPECT_THROW(std::ignore = builder.build(0), std::runtime_error);
}

TEST_F(ChunkTest, ChunkBuilderMixedMessages) {
    ChunkID chunk_id = 123;
    ChunkBuilder builder(stream, br.get(), 6);  // Hint for 7 messages

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 4, 5, 6};  // Concatenated metadata
    std::vector<uint8_t> data{6, 7, 8, 9, 10};  // Concatenated data

    // Add messages in sequence
    builder.add_control_message(1, 10);  // control message 1
    builder.add_packed_data(
        2, create_packed_data({metadata.data(), 3}, {data.data(), 3}, stream)
    );  // packed data 1
    builder.add_control_message(3, 40);  // control message 2
    builder.add_packed_data(
        4, create_packed_data({metadata.data() + 5, 0}, {data.data() + 5, 0}, stream)
    );  // empty packed data - non-null
    builder.add_packed_data(
        5, create_packed_data({metadata.data() + 3, 2}, {data.data() + 3, 2}, stream)
    );  // packed data 2
    builder.add_packed_data(
        6,
        PackedData{
            std::make_unique<std::vector<uint8_t>>(metadata.begin() + 5, metadata.end()),
            nullptr
        }
    );  // metadata only packed data

    // Add empty packed data - null should throw a logic error
    EXPECT_THROW(
        builder.add_packed_data(7, PackedData{nullptr, nullptr}), std::logic_error
    );

    auto chunk = builder.build(chunk_id);

    // Verify the chunk properties
    EXPECT_EQ(chunk.chunk_id(), chunk_id);
    EXPECT_EQ(chunk.n_messages(), 6);

    // Helper function to verify message properties
    auto test_message = [&](size_t i,
                            PartID part_id,
                            size_t expected_chunks,
                            bool is_control,
                            uint32_t meta_size,
                            size_t data_size) {
        EXPECT_EQ(chunk.part_id(i), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(i), expected_chunks);
        EXPECT_EQ(chunk.is_control_message(i), is_control);
        EXPECT_EQ(chunk.metadata_size(i), meta_size);
        EXPECT_EQ(chunk.data_size(i), data_size);

        auto chunk_copy = chunk.get_data(chunk_id, i, stream, br.get());

        EXPECT_EQ(chunk_copy.part_id(0), part_id);
        EXPECT_EQ(chunk_copy.expected_num_chunks(0), expected_chunks);
        EXPECT_EQ(chunk_copy.is_control_message(0), is_control);
        EXPECT_EQ(chunk_copy.metadata_size(0), meta_size);
        EXPECT_EQ(chunk_copy.data_size(0), data_size);
    };

    // Verify each message
    test_message(0, 1, 10, true, 0, 0);  // control message 1
    test_message(1, 2, 0, false, 3, 3);  // packed data 1
    test_message(2, 3, 40, true, 0, 0);  // control message 2
    test_message(3, 4, 0, false, 0, 0);  // empty packed data - non-null
    test_message(4, 5, 0, false, 2, 2);  // packed data 2
    test_message(5, 6, 0, false, 1, 0);  // metadata only packed data

    // Release and verify buffers
    auto released_metadata = chunk.release_metadata_buffer();
    auto released_data = chunk.release_data_buffer();

    // Verify metadata buffer
    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(metadata, *released_metadata);

    // Verify data buffer
    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, data.size());  // Total size of data
    std::vector<uint8_t> host_data(data.size());
    RAPIDSMPF_CUDA_TRY(cudaMemcpy(
        host_data.data(), released_data->data(), data.size(), cudaMemcpyDeviceToHost
    ));
    EXPECT_EQ(data, host_data);

    // after building, the builder is empty
    EXPECT_THROW(std::ignore = builder.build(0), std::runtime_error);
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
    ChunkBuilder builder(stream, br.get(), 2);

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data

    auto chunk = builder.add_packed_data(1, create_packed_data(metadata, data, stream))
                     .build(chunk_id);

    EXPECT_EQ(MemoryType::HOST, chunk.data_memory_type());
}
