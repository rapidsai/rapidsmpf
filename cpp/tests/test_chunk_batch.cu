/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <thrust/equal.h>

#include <rmm/exec_policy.hpp>

#include <rapidsmpf/shuffler/chunk_batch.hpp>

// NOTE: this test is in a .cu file, because it uses thrust::equal which requires nvcc.

namespace rapidsmpf::shuffler::detail {

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

        // iterators won't advance
        EXPECT_EQ(batch.end(stream), batch.begin(stream));
    };

    auto batch1 = ChunkBatch::create(id, rank, {}, &br, stream);
    test_empty_batch(batch1);

    // release the metadata buffer
    auto metadata = batch1.release_metadata();
    EXPECT_EQ(ChunkBatch::batch_header_size, metadata->size());

    auto batch2 = ChunkBatch::create(std::move(metadata), {});
    test_empty_batch(batch2);

    // create chunk batch with in invalid metadata buffer -> should throw
    EXPECT_THROW(
        std::ignore = ChunkBatch::create(std::make_unique<std::vector<uint8_t>>(), {}),
        std::logic_error
    );
}

/**
 * Types of chunks
 * 1. Chunks with control messages
 * 2. Chunks with no data
 * 3. Chunks with data
 *    a. Device data
 *    b. Host data
 */

// Parametarized test for MemoryType and types of chunks
class ChunkBatchTest
    : public ::testing::TestWithParam<std::tuple<MemoryType, std::string>> {
  public:
    // dummy data for buffers
    static constexpr std::initializer_list<uint8_t> dummy_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    static constexpr std::ptrdiff_t len = dummy_data.size();

    // dummy batch details
    static constexpr uint32_t id = 1;
    static constexpr Rank rank = 2;

    // dummy data buffer
    std::unique_ptr<Buffer> data_buf;

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    BufferResource br{cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, [] {
                                                                     return 1000_KiB;
                                                                 }}}};

    ChunkBatchTest() {
        data_buf = allocate_buffer(get_memory_type(), len, br, stream);
        // copy dummy data to the buffer
        switch (get_memory_type()) {
        case MemoryType::DEVICE:
            RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                data_buf->data(),
                std::data(dummy_data),
                len,
                cudaMemcpyHostToDevice,
                stream
            ));
            break;
        case MemoryType::HOST:
            std::memcpy(data_buf->data(), std::data(dummy_data), len);
            break;
        }
    }

    MemoryType get_memory_type() const {
        return std::get<0>(GetParam());
    }

    auto copy_metadata() const {
        return std::make_unique<std::vector<uint8_t>>(dummy_data);
    }

    auto copy_data() const {
        return data_buf->copy_slice(0, len, stream);
    }

    auto gen_chunks() const {
        std::vector<Chunk> chunks;

        auto const& chunks_type = std::get<1>(GetParam());
        if (chunks_type == "mixed") {
            chunks.emplace_back(1, 1, 0, len, copy_metadata(), copy_data());
            chunks.emplace_back(2, 2, 0, len, copy_metadata(), copy_data());
            chunks.emplace_back(3, 3, 100);
            chunks.emplace_back(4, 4, 0, len, copy_metadata(), copy_data());
            chunks.emplace_back(5, 5, 101);
            chunks.emplace_back(6, 6, 0, len, copy_metadata(), copy_data());
        } else if (chunks_type == "no_data") {
            chunks.emplace_back(3, 3, 100);
            chunks.emplace_back(5, 5, 101);
        }
        // else -> empty chunks

        return chunks;
    }

    void test_batch(std::vector<Chunk> const& exp_chunks, ChunkBatch const& batch) const {
        EXPECT_EQ(id, batch.id());
        EXPECT_EQ(rank, batch.destination());
        EXPECT_EQ(exp_chunks.size(), batch.size());

        // std::vector<Chunk> const chunks = batch.get_chunks(stream);
        // EXPECT_EQ(exp_chunks.size(), chunks.size());

        auto chunk = batch.begin(stream);
        for (size_t i = 0; i < exp_chunks.size(); i++, chunk++) {
            SCOPED_TRACE("chunk " + std::to_string(i));
            EXPECT_EQ(exp_chunks[i].pid, chunk->pid);
            EXPECT_EQ(exp_chunks[i].cid, chunk->cid);
            EXPECT_EQ(exp_chunks[i].expected_num_chunks, chunk->expected_num_chunks);
            EXPECT_EQ(exp_chunks[i].gpu_data_size, chunk->gpu_data_size);

            if (exp_chunks[i].metadata) {
                SCOPED_TRACE("chunk metadata" + std::to_string(i));
                EXPECT_EQ(*exp_chunks[i].metadata, *chunk->metadata);
            }

            if (exp_chunks[i].gpu_data) {
                switch (get_memory_type()) {
                case MemoryType::DEVICE:
                    {
                        SCOPED_TRACE("chunk data device" + std::to_string(i));
                        EXPECT_TRUE(
                            thrust::equal(
                                rmm::exec_policy(stream, br.device_mr()),
                                static_cast<cuda::std::byte*>(
                                    exp_chunks[i].gpu_data->data()
                                ),
                                static_cast<cuda::std::byte*>(
                                    exp_chunks[i].gpu_data->data()
                                ) + len,
                                static_cast<cuda::std::byte*>(chunk->gpu_data->data())
                            )
                        );
                        break;
                    }
                case MemoryType::HOST:
                    {
                        SCOPED_TRACE("chunk data host" + std::to_string(i));
                        EXPECT_EQ(
                            *(const_cast<Buffer const&>(*exp_chunks[i].gpu_data).host()),
                            *(const_cast<Buffer const&>(*chunk->gpu_data).host())
                        );
                        break;
                    }
                }
            }
        }
        EXPECT_EQ(batch.end(stream), chunk);  // check if the iterator has reached the end
    };
};

INSTANTIATE_TEST_SUITE_P(
    ChunkBatchTestP,
    ChunkBatchTest,
    ::testing::Combine(
        ::testing::Values(MemoryType::DEVICE, MemoryType::HOST),
        ::testing::Values("mixed", "no_data", "empty")
    ),
    [](const ::testing::TestParamInfo<ChunkBatchTest::ParamType>& info) {
        return std::string(
                   std::get<0>(info.param) == MemoryType::HOST ? "host" : "device"
               )
               + "__" + std::get<1>(info.param);
    }
);

TEST_P(ChunkBatchTest, Run) {
    auto exp_chunks = gen_chunks();

    auto batch1 = ChunkBatch::create(id, rank, gen_chunks(), &br, stream);
    EXPECT_NO_FATAL_FAILURE(test_batch(exp_chunks, batch1));

    // reacreate a new batch using the first batch
    auto batch2 = ChunkBatch::create(batch1.release_metadata(), batch1.release_payload());
    EXPECT_NO_FATAL_FAILURE(test_batch(exp_chunks, batch2));
}

}  // namespace rapidsmpf::shuffler::detail
