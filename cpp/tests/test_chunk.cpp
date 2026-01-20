/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    EXPECT_THROW(chunk3.get_data(chunk_id, 1, br.get()), std::out_of_range);
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

TEST_F(ChunkTest, ChunkConcatControlMessages) {
    ChunkID chunk_id = 123;
    std::vector<Chunk> chunks;

    // Create three control message chunks
    chunks.push_back(Chunk::from_finished_partition(0, 1, 10));
    chunks.push_back(Chunk::from_finished_partition(0, 2, 20));
    chunks.push_back(Chunk::from_finished_partition(0, 3, 30));

    auto concat_chunk = Chunk::concat(std::move(chunks), chunk_id, br.get());

    // Verify the concatenated chunk properties
    EXPECT_EQ(concat_chunk.chunk_id(), chunk_id);
    EXPECT_EQ(concat_chunk.n_messages(), 3);

    // Verify each message in the concatenated chunk
    auto test_message = [&](size_t i, PartID part_id, size_t expected_chunks) {
        EXPECT_EQ(concat_chunk.part_id(i), part_id);
        EXPECT_EQ(concat_chunk.expected_num_chunks(i), expected_chunks);
        EXPECT_TRUE(concat_chunk.is_control_message(i));
        EXPECT_EQ(concat_chunk.metadata_size(i), 0);
        EXPECT_EQ(concat_chunk.data_size(i), 0);
    };

    test_message(0, 1, 10);
    test_message(1, 2, 20);
    test_message(2, 3, 30);
}

TEST_F(ChunkTest, ChunkConcatPackedData) {
    ChunkID chunk_id = 123;
    std::vector<Chunk> chunks;

    // Create test metadata and data as single concatenated vectors
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data

    // Create two chunks with packed data using spans
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            1,
            create_packed_data({metadata.data(), 3}, {data.data(), 3}, stream, br.get())
        )
    );
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            2,
            create_packed_data(
                {metadata.data() + 3, 2}, {data.data() + 3, 2}, stream, br.get()
            )
        )
    );

    auto concat_chunk = Chunk::concat(std::move(chunks), chunk_id, br.get());

    // Verify the concatenated chunk properties
    EXPECT_EQ(concat_chunk.chunk_id(), chunk_id);
    EXPECT_EQ(concat_chunk.n_messages(), 2);

    // Verify each message in the concatenated chunk
    auto test_message =
        [&](size_t i, PartID part_id, size_t meta_size, size_t data_size) {
            EXPECT_EQ(concat_chunk.part_id(i), part_id);
            EXPECT_EQ(concat_chunk.expected_num_chunks(i), 0);
            EXPECT_FALSE(concat_chunk.is_control_message(i));
            EXPECT_EQ(concat_chunk.metadata_size(i), meta_size);
            EXPECT_EQ(concat_chunk.data_size(i), data_size);
        };

    test_message(0, 1, 3, 3);
    test_message(1, 2, 2, 2);

    // Verify the concatenated metadata and data
    auto released_metadata = concat_chunk.release_metadata_buffer();
    auto released_data = concat_chunk.release_data_buffer();
    released_data->stream().synchronize();

    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(released_metadata->size(), 5);  // Total size of both metadata chunks
    EXPECT_EQ(*released_metadata, metadata);

    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, 5);  // Total size of both data chunks
    std::vector<uint8_t> host_data(5);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(host_data.data(), released_data->data(), 5, cudaMemcpyDeviceToHost)
    );
    EXPECT_EQ(host_data, data);
}

std::tuple<Chunk, std::vector<uint8_t>, std::vector<uint8_t>, size_t> make_mixed_chunk(
    ChunkID chunk_id,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    PartID part_id_offset = 0
) {
    std::vector<Chunk> chunks;

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 4, 5, 6};
    std::vector<uint8_t> data{6, 7, 8, 9, 10};

    // Create chunks with mixed message types
    chunks.push_back(
        Chunk::from_finished_partition(0, 1 + part_id_offset, 10)
    );  // control message
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            2 + part_id_offset,
            create_packed_data({metadata.data(), 3}, {data.data(), 3}, stream, br)
        )
    );  // packed data
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            3 + part_id_offset,
            create_packed_data({metadata.data() + 5, 0}, {data.data() + 5, 0}, stream, br)
        )
    );  // empty packed data - non-null
    chunks.push_back(
        Chunk::from_finished_partition(0, 4 + part_id_offset, 20)
    );  // control message
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            5 + part_id_offset,
            create_packed_data({metadata.data() + 3, 2}, {data.data() + 3, 2}, stream, br)
        )
    );  // packed data
    chunks.push_back(
        Chunk::from_packed_data(
            0,
            6 + part_id_offset,
            create_packed_data(
                {metadata.begin() + 5, metadata.end()}, {data.data(), 0}, stream, br
            )
        )
    );  // metadata only packed data

    return std::make_tuple(
        Chunk::concat(std::move(chunks), chunk_id, br), metadata, data, 6
    );
}

TEST_F(ChunkTest, ChunkConcatMixedMessages) {
    ChunkID chunk_id = 123;

    auto [concat_chunk, metadata, data, count] =
        make_mixed_chunk(chunk_id, stream, br.get());

    // Verify the concatenated chunk properties
    EXPECT_EQ(concat_chunk.chunk_id(), chunk_id);
    EXPECT_EQ(concat_chunk.n_messages(), count);

    // Verify each message in the concatenated chunk
    auto test_message = [&](size_t i,
                            PartID part_id,
                            size_t expected_chunks,
                            bool is_control,
                            uint32_t meta_size,
                            size_t data_size) {
        SCOPED_TRACE("test_message: " + std::to_string(i));
        EXPECT_EQ(concat_chunk.part_id(i), part_id);
        EXPECT_EQ(concat_chunk.expected_num_chunks(i), expected_chunks);
        EXPECT_EQ(concat_chunk.is_control_message(i), is_control);
        EXPECT_EQ(concat_chunk.metadata_size(i), meta_size);
        EXPECT_EQ(concat_chunk.data_size(i), data_size);

        auto chunk_copy = concat_chunk.get_data(chunk_id, i, br.get());

        EXPECT_EQ(chunk_copy.part_id(0), part_id);
        EXPECT_EQ(chunk_copy.expected_num_chunks(0), expected_chunks);
        EXPECT_EQ(chunk_copy.is_control_message(0), is_control);
        EXPECT_EQ(chunk_copy.metadata_size(0), meta_size);
        EXPECT_EQ(chunk_copy.data_size(0), data_size);
    };

    test_message(0, 1, 10, true, 0, 0);  // control message
    test_message(1, 2, 0, false, 3, 3);  // packed data
    test_message(2, 3, 0, false, 0, 0);  // empty packed data - non-null
    test_message(3, 4, 20, true, 0, 0);  // control message
    test_message(4, 5, 0, false, 2, 2);  // packed data
    test_message(5, 6, 0, false, 1, 0);  // metadata only packed data

    // Verify the concatenated metadata and data
    auto released_metadata = concat_chunk.release_metadata_buffer();
    auto released_data = concat_chunk.release_data_buffer();
    released_data->stream().synchronize();

    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(released_metadata->size(), 6);  // Total size of metadata
    EXPECT_EQ(*released_metadata, metadata);

    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, 5);  // Total size of data
    std::vector<uint8_t> host_data(5);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(host_data.data(), released_data->data(), 5, cudaMemcpyDeviceToHost)
    );
    EXPECT_EQ(host_data, data);
}

TEST_F(ChunkTest, ChunkConcatMixedMessagesMultiple) {
    auto [concat_chunk1, metadata1, data1, count1] =
        make_mixed_chunk(0, stream, br.get());
    auto [concat_chunk2, metadata2, data2, count2] =
        make_mixed_chunk(1, stream, br.get(), static_cast<PartID>(count1));

    std::vector<Chunk> chunks;
    chunks.push_back(std::move(concat_chunk1));
    chunks.push_back(std::move(concat_chunk2));

    auto concat_chunk = Chunk::concat(std::move(chunks), 2, br.get());

    // Verify the concatenated chunk properties
    EXPECT_EQ(concat_chunk.chunk_id(), 2);
    EXPECT_EQ(concat_chunk.n_messages(), count1 + count2);

    // Verify the concatenated metadata and data
    auto released_metadata = concat_chunk.release_metadata_buffer();
    auto released_data = concat_chunk.release_data_buffer();
    released_data->stream().synchronize();

    ASSERT_NE(released_metadata, nullptr);
    // Total size of metadata
    EXPECT_EQ(released_metadata->size(), metadata1.size() + metadata2.size());
    EXPECT_TRUE(
        std::equal(metadata1.begin(), metadata1.end(), released_metadata->begin())
    );
    EXPECT_TRUE(
        std::equal(
            metadata2.begin(),
            metadata2.end(),
            released_metadata->begin() + metadata1.size()
        )
    );

    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, data1.size() + data2.size());  // Total size of data
    std::vector<uint8_t> host_data(data1.size() + data2.size());
    RAPIDSMPF_CUDA_TRY(cudaMemcpy(
        host_data.data(),
        released_data->data(),
        data1.size() + data2.size(),
        cudaMemcpyDeviceToHost
    ));
    EXPECT_TRUE(std::equal(data1.begin(), data1.end(), host_data.begin()));
    EXPECT_TRUE(std::equal(data2.begin(), data2.end(), host_data.begin() + data1.size()));
}

TEST_F(ChunkTest, ChunkConcatSingleChunk) {
    ChunkID chunk_id = 123;
    std::vector<Chunk> chunks;

    // Create a single chunk with packed data
    std::vector<uint8_t> metadata{1, 2, 3};
    std::vector<uint8_t> data{4, 5, 6};

    auto packed_data = create_packed_data(metadata, data, stream, br.get());
    auto expected_metadata_ptr = packed_data.metadata->data();
    auto expected_data_ptr = packed_data.data->data();

    chunks.push_back(Chunk::from_packed_data(0, 1, std::move(packed_data)));

    auto concat_chunk = Chunk::concat(std::move(chunks), chunk_id, br.get());

    // Verify the concatenated chunk properties
    EXPECT_EQ(concat_chunk.chunk_id(), chunk_id);
    EXPECT_EQ(concat_chunk.n_messages(), 1);

    // Verify the message in the concatenated chunk
    EXPECT_EQ(concat_chunk.part_id(0), 1);
    EXPECT_EQ(concat_chunk.expected_num_chunks(0), 0);
    EXPECT_FALSE(concat_chunk.is_control_message(0));
    EXPECT_EQ(concat_chunk.metadata_size(0), 3);
    EXPECT_EQ(concat_chunk.data_size(0), 3);

    // Verify the metadata and data
    auto released_metadata = concat_chunk.release_metadata_buffer();
    auto released_data = concat_chunk.release_data_buffer();
    released_data->stream().synchronize();

    ASSERT_NE(released_metadata, nullptr);
    EXPECT_EQ(*released_metadata, metadata);
    // verify the metadata pointer is the same as the original pointer
    EXPECT_EQ(released_metadata->data(), expected_metadata_ptr);

    ASSERT_NE(released_data, nullptr);
    EXPECT_EQ(released_data->size, 3);
    std::vector<uint8_t> host_data(3);
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(host_data.data(), released_data->data(), 3, cudaMemcpyDeviceToHost)
    );
    EXPECT_EQ(host_data, data);
    // verify the data pointer is the same as the original pointer
    EXPECT_EQ(released_data->data(), expected_data_ptr);
}

TEST_F(ChunkTest, ChunkConcatEmptyVector) {
    std::vector<Chunk> chunks;
    EXPECT_THROW(Chunk::concat(std::move(chunks), 123, br.get()), std::logic_error);
}

TEST_F(ChunkTest, ChunkConcatHostBufferAllocation) {
    // create a new buffer resource with only host memory available
    br = std::make_unique<BufferResource>(
        cudf::get_current_device_resource_ref(),
        PinnedMemoryResource::Disabled,
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{
            {MemoryType::DEVICE, []() { return 0; }}
        }
    );

    ChunkID chunk_id = 123;

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data

    // create two chunks with packed data -> this should concatenate the two chunks into a
    // single chunk
    std::vector<Chunk> chunks;
    chunks.push_back(
        Chunk::from_packed_data(
            1, 1, create_packed_data(metadata, data, stream, br.get())
        )
    );
    chunks.push_back(
        Chunk::from_packed_data(
            2, 2, create_packed_data(metadata, data, stream, br.get())
        )
    );
    auto chunk = Chunk::concat(std::move(chunks), chunk_id, br.get());

    EXPECT_EQ(MemoryType::HOST, chunk.data_memory_type());
}

TEST_F(ChunkTest, ChunkConcatPreferredMemoryType) {
    ChunkID chunk_id = 123;

    // Create test metadata and data
    std::vector<uint8_t> metadata{1, 2, 3, 7, 8};  // Concatenated metadata
    std::vector<uint8_t> data{4, 5, 6, 9, 10};  // Concatenated data
    auto gen_chunks = [&] {
        std::vector<Chunk> chunks;
        chunks.push_back(
            Chunk::from_packed_data(
                1, 1, create_packed_data(metadata, data, stream, br.get())
            )
        );
        chunks.push_back(
            Chunk::from_packed_data(
                2, 2, create_packed_data(metadata, data, stream, br.get())
            )
        );
        return chunks;
    };

    // test with both memory types
    for (auto mem_type : {MemoryType::HOST, MemoryType::DEVICE}) {
        SCOPED_TRACE("mem_type: " + std::to_string(static_cast<int>(mem_type)));
        auto chunk = Chunk::concat(gen_chunks(), chunk_id, br.get(), mem_type);
        EXPECT_EQ(mem_type, chunk.data_memory_type());
    }
}
