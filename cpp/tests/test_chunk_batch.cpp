/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <rapidsmp/shuffler/chunk_batch.hpp>

namespace rapidsmp::shuffler::detail {

constexpr std::size_t operator""_KiB(unsigned long long n) {
    return n * (1 << 10);
}

// Allocate a buffer of the given size from the given resource.
std::unique_ptr<Buffer> allocate_buffer(
    MemoryType mem_type,
    std::size_t size,
    BufferResource& br,
    rmm::cuda_stream_view stream
) {
    auto [res, _] = br.reserve(mem_type, size, false);
    return br.allocate(mem_type, size, stream, res);
}

TEST(ChunkBatch, Empty) {
    uint32_t id = 1;
    Rank rank = 2;
    auto stream = rmm::cuda_stream_default;
    auto dev_mem_available = []() -> std::int64_t { return 1000_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, dev_mem_available}}
    };

    auto test_empty_batch = [&](ChunkBatch& batch) {
        EXPECT_EQ(id, batch.id());
        EXPECT_EQ(rank, batch.destination());
        EXPECT_EQ(0, batch.size());

        auto chunks = batch.get_chunks(stream);
        EXPECT_EQ(0, chunks.size());
    };

    auto batch1 = ChunkBatch::create(id, rank, {}, &br, stream);
    test_empty_batch(batch1);

    // release the metadata buffer
    auto metadata = batch1.release_metadata();
    EXPECT_EQ(ChunkBatch::batch_header_size, metadata->size());

    auto batch2 = ChunkBatch::create(std::move(metadata), {});
    test_empty_batch(batch2);
}

/**
 * Types of chunks
 * 1. Chunks with control messages
 * 2. Chunks with no data
 * 3. Chunks with data
 *    a. Device data
 *    b. Host data
 */

// Parametarized test for MemoryType
class ChunkBatchTest : public ::testing::TestWithParam<MemoryType> {
  public:
    // dummy metadata buffer
    std::vector<uint8_t> const dummy_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    // dummy data buffer
    std::unique_ptr<Buffer> data;

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    std::ptrdiff_t const len = dummy_data.size();

    BufferResource br{cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, [] {
                                                                     return 1000_KiB;
                                                                 }}}};

    ChunkBatchTest() {
        data = allocate_buffer(GetParam(), dummy_data.size(), br, stream);
        if (GetParam() == MemoryType::HOST) {
            std::memcpy(data->data(), dummy_data.data(), dummy_data.size());
        } else {
            RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                data->data(),
                dummy_data.data(),
                dummy_data.size(),
                cudaMemcpyHostToDevice,
                stream
            ));
        }
    }

    auto copy_metadata() const {
        return std::make_unique<std::vector<uint8_t>>(dummy_data);
    }

    auto copy_data() const {
        return data->copy_slice(0, len, stream);
    }

    auto make_chunks() const {
        std::vector<Chunk> chunks{};
        chunks.emplace_back(1, 1, 0, len, copy_metadata(), copy_data());
        chunks.emplace_back(2, 2, 0, len, copy_metadata(), copy_data());
        chunks.emplace_back(3, 3, 100);
        chunks.emplace_back(4, 4, 0, len, copy_metadata(), copy_data());
        chunks.emplace_back(5, 5, 100);
        chunks.emplace_back(6, 6, 0, len, copy_metadata(), copy_data());

        return chunks;
    }
};

INSTANTIATE_TEST_SUITE_P(
    ChunkBatchTestP,
    ChunkBatchTest,
    ::testing::Values(MemoryType::DEVICE, MemoryType::HOST),
    [](const ::testing::TestParamInfo<ChunkBatchTest::ParamType>& info) {
        return info.param == MemoryType::HOST ? "host" : "device";
    }
);

TEST_P(ChunkBatchTest, NonEmptyDeviceData) {
    uint32_t id = 1;
    Rank rank = 2;

    std::vector<Chunk> exp_chunks = make_chunks();

    auto test_batch = [&](auto const& batch) {
        EXPECT_EQ(id, batch.id());
        EXPECT_EQ(rank, batch.destination());
        EXPECT_EQ(exp_chunks.size(), batch.size());

        std::vector<Chunk> const chunks = batch.get_chunks(stream);
        EXPECT_EQ(exp_chunks.size(), chunks.size());

        for (size_t i = 0; i < exp_chunks.size(); i++) {
            SCOPED_TRACE(i);
            EXPECT_EQ(exp_chunks[i].pid, chunks[i].pid);
            EXPECT_EQ(exp_chunks[i].cid, chunks[i].cid);
            EXPECT_EQ(exp_chunks[i].expected_num_chunks, chunks[i].expected_num_chunks);
            EXPECT_EQ(exp_chunks[i].gpu_data_size, chunks[i].gpu_data_size);

            if (exp_chunks[i].metadata) {
                EXPECT_EQ(*exp_chunks[i].metadata, *chunks[i].metadata);
            }

            if (exp_chunks[i].gpu_data) {
                switch (GetParam()) {
                case MemoryType::DEVICE:
                    {
                        auto host_copy = chunks[i].gpu_data->copy_slice(
                            MemoryType::HOST, 0, len, stream
                        );
                        cudaStreamSynchronize(stream);
                        EXPECT_EQ(
                            0, std::memcmp(dummy_data.data(), host_copy->data(), len)
                        );
                        break;
                    }
                case MemoryType::HOST:
                    {
                        EXPECT_EQ(
                            *(const_cast<Buffer const&>(*exp_chunks[i].gpu_data).host()),
                            *(const_cast<Buffer const&>(*chunks[i].gpu_data).host())
                        );
                        EXPECT_EQ(
                            0,
                            std::memcmp(
                                dummy_data.data(), chunks[i].gpu_data->data(), len
                            )
                        );
                        break;
                    }
                }
            }
        }
    };

    auto batch1 = ChunkBatch::create(id, rank, make_chunks(), &br, stream);
    test_batch(batch1);

    auto batch2 = ChunkBatch::create(batch1.release_metadata(), batch1.release_payload());
    test_batch(batch2);
}


}  // namespace rapidsmp::shuffler::detail
