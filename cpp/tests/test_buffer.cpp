/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <cuda/memory>

#include <cudf_test/base_fixture.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

namespace {

// unlike cudaMemcpyAsync, cudaMemsetAsync does not transparently handle host ptrs on all
// architectures.
void checked_memset(
    void* ptr, std::size_t size, std::uint8_t value, rmm::cuda_stream_view stream
) {
    if (cuda::is_device_accessible(ptr, rmm::get_current_cuda_device().value())) {
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, value, size, stream));
    } else {
        std::memset(ptr, value, size);
    }
}

}  // namespace

class BufferRebindStreamTest : public ::testing::TestWithParam<MemoryType> {
  protected:
    void SetUp() override {
        stream_pool = std::make_shared<rmm::cuda_stream_pool>(2);

        if (GetParam() == MemoryType::PINNED_HOST
            && !is_pinned_memory_resources_supported())
        {
            GTEST_SKIP() << "Pinned memory resources are not supported on this system";
        }

        br = std::make_unique<BufferResource>(
            cudf::get_current_device_resource_ref(),
            PinnedMemoryResource::make_fixed_sized_if_available(get_current_numa_node()),
            std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{},
            std::nullopt,
            stream_pool
        );

        std::mt19937 rng(42);
        std::uniform_int_distribution<std::uint8_t> dist(0, 255);
        random_data.resize(buffer_size);
        for (auto& byte : random_data) {
            byte = dist(rng);
        }
    }

    static constexpr std::size_t buffer_size = 32_MiB;
    static constexpr std::size_t chunk_size = 1_MiB;

    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    std::unique_ptr<BufferResource> br;
    std::vector<std::uint8_t> random_data;
};

INSTANTIATE_TEST_SUITE_P(
    MemoryTypes,
    BufferRebindStreamTest,
    ::testing::Values(MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST),
    [](const ::testing::TestParamInfo<MemoryType>& info) { return to_string(info.param); }
);

TEST_P(BufferRebindStreamTest, RebindStreamAndCopy) {
    MemoryType mem_type = GetParam();
    if (mem_type == MemoryType::PINNED_HOST) {
        GTEST_SKIP() << "TODO reenable this test";
    }
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    auto rmm_buffer = std::make_unique<rmm::device_buffer>(
        random_data.data(), buffer_size, stream1, br->device_mr()
    );

    auto [reserve1, overbooking1] =
        br->reserve(mem_type, buffer_size, AllowOverbooking::YES);
    auto buffer1 = br->allocate(buffer_size, stream1, reserve1);
    EXPECT_EQ(buffer1->mem_type(), mem_type);
    EXPECT_EQ(buffer1->stream().value(), stream1.value());

    std::size_t num_chunks = buffer_size / chunk_size;
    for (std::size_t i = 0; i < num_chunks; ++i) {
        std::size_t offset = i * chunk_size;
        buffer1->write_access([&](std::byte* dst, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                dst + offset,
                static_cast<std::byte*>(rmm_buffer->data()) + offset,
                chunk_size,
                cudaMemcpyDefault,
                stream
            ));
        });
    }

    auto [reserve2, overbooking2] =
        br->reserve(mem_type, buffer_size, AllowOverbooking::YES);
    auto buffer2 = br->allocate(buffer_size, stream2, reserve2);
    EXPECT_EQ(buffer2->mem_type(), mem_type);
    EXPECT_EQ(buffer2->stream().value(), stream2.value());

    buffer1->rebind_stream(stream2);
    EXPECT_EQ(buffer1->stream().value(), stream2.value());

    buffer_copy(br->statistics(), *buffer2, *buffer1, buffer_size);

    std::vector<std::uint8_t> result(buffer_size);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        result.data(), buffer2->data(), buffer_size, cudaMemcpyDefault, stream2
    ));
    stream2.synchronize();

    EXPECT_EQ(result, random_data);
}

TEST_P(BufferRebindStreamTest, RebindStreamSynchronizesCorrectly) {
    MemoryType mem_type = GetParam();
    if (mem_type == MemoryType::PINNED_HOST) {
        GTEST_SKIP() << "TODO reenable this test";
    }
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    constexpr std::size_t test_size = 4_MiB;

    auto [reserve1, overbooking1] =
        br->reserve(mem_type, test_size, AllowOverbooking::YES);
    auto buffer1 = br->allocate(test_size, stream1, reserve1);
    EXPECT_EQ(buffer1->mem_type(), mem_type);

    buffer1->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        checked_memset(ptr, test_size, 0xAB, stream);
    });

    buffer1->rebind_stream(stream2);
    EXPECT_EQ(buffer1->stream().value(), stream2.value());

    buffer1->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        checked_memset(ptr, test_size / 2, 0xCD, stream);
    });

    std::vector<std::uint8_t> result(test_size);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        result.data(), buffer1->data(), test_size, cudaMemcpyDefault, stream2
    ));
    stream2.synchronize();

    for (std::size_t i = 0; i < test_size / 2; ++i) {
        EXPECT_EQ(result[i], 0xCD) << "Mismatch at index " << i;
    }
    for (std::size_t i = test_size / 2; i < test_size; ++i) {
        EXPECT_EQ(result[i], 0xAB) << "Mismatch at index " << i;
    }
}

TEST_P(BufferRebindStreamTest, MultipleRebinds) {
    MemoryType mem_type = GetParam();
    if (mem_type == MemoryType::PINNED_HOST) {
        GTEST_SKIP() << "TODO reenable this test";
    }
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    constexpr std::size_t test_size = 2_MiB;
    auto [reserve, overbooking] = br->reserve(mem_type, test_size, AllowOverbooking::YES);
    auto buffer = br->allocate(test_size, stream1, reserve);
    EXPECT_EQ(buffer->mem_type(), mem_type);

    buffer->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        checked_memset(ptr, test_size, 0x11, stream);
    });

    buffer->rebind_stream(stream2);
    EXPECT_EQ(buffer->stream().value(), stream2.value());
    buffer->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        checked_memset(ptr, test_size / 2, 0x22, stream);
    });

    buffer->rebind_stream(stream1);
    EXPECT_EQ(buffer->stream().value(), stream1.value());
    buffer->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        checked_memset(ptr + test_size / 2, test_size / 2, 0x33, stream);
    });

    std::vector<std::uint8_t> result(test_size);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        result.data(), buffer->data(), test_size, cudaMemcpyDefault, stream1
    ));
    stream1.synchronize();

    for (std::size_t i = 0; i < test_size / 2; ++i) {
        EXPECT_EQ(result[i], 0x22) << "Mismatch at index " << i;
    }
    for (std::size_t i = test_size / 2; i < test_size; ++i) {
        EXPECT_EQ(result[i], 0x33) << "Mismatch at index " << i;
    }
}

TEST_P(BufferRebindStreamTest, ThrowsWhenLocked) {
    MemoryType mem_type = GetParam();
    if (mem_type == MemoryType::PINNED_HOST) {
        GTEST_SKIP() << "TODO reenable this test";
    }
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    constexpr std::size_t test_size = 1_MiB;
    auto [reserve, overbooking] = br->reserve(mem_type, test_size, AllowOverbooking::YES);
    auto buffer = br->allocate(test_size, stream1, reserve);
    EXPECT_EQ(buffer->mem_type(), mem_type);

    auto* ptr = buffer->exclusive_data_access();
    EXPECT_NE(ptr, nullptr);
    EXPECT_THROW(buffer->rebind_stream(stream2), std::logic_error);
    buffer->unlock();
    EXPECT_NO_THROW(buffer->rebind_stream(stream2));
    EXPECT_EQ(buffer->stream().value(), stream2.value());

    EXPECT_NO_THROW(buffer->rebind_stream(stream2));
    EXPECT_EQ(buffer->stream().value(), stream2.value());
}

// =============================================================================
// Buffer::copy_to test suite
// =============================================================================

namespace {

/**
 * @brief Identifies the memory kind of a buffer for parameterized copy_to tests.
 *
 * PINNED_64 and PINNED_128 both map to MemoryType::PINNED_HOST but use different
 * fixed-size block sizes (64 B and 128 B respectively). Two separate BufferResources
 * are used per test because a BufferResource may only hold one PinnedMemoryResource.
 */
enum class BufferKind {
    DEVICE,
    HOST,
    PINNED_64,
    PINNED_128
};

std::string_view buffer_kind_to_string(BufferKind kind) noexcept {
    switch (kind) {
    case BufferKind::DEVICE:
        return "DEVICE";
    case BufferKind::HOST:
        return "HOST";
    case BufferKind::PINNED_64:
        return "PINNED64";
    case BufferKind::PINNED_128:
        return "PINNED128";
    }
    return "UNKNOWN";
}

MemoryType to_memory_type(BufferKind kind) noexcept {
    switch (kind) {
    case BufferKind::DEVICE:
        return MemoryType::DEVICE;
    case BufferKind::HOST:
        return MemoryType::HOST;
    case BufferKind::PINNED_64:
    case BufferKind::PINNED_128:
        return MemoryType::PINNED_HOST;
    }
    return MemoryType::HOST;
}

bool kind_needs_pinned(BufferKind kind) noexcept {
    return kind == BufferKind::PINNED_64 || kind == BufferKind::PINNED_128;
}

struct CopyToParam {
    BufferKind src_kind;
    BufferKind dst_kind;
    std::size_t copy_size;
    std::ptrdiff_t src_offset;
    std::ptrdiff_t dst_offset;
};

std::shared_ptr<BufferResource> make_copy_test_br(
    BufferKind kind, std::shared_ptr<rmm::cuda_stream_pool> pool
) {
    std::shared_ptr<PinnedMemoryResource> pinned_mr = PinnedMemoryResource::Disabled;
    // 1 MiB pool is ample for the 1 KiB buffers used in these tests.
    PinnedPoolProperties pool_properties{
        .initial_pool_size = 1_MiB, .max_pool_size = 1_MiB
    };
    if (kind == BufferKind::PINNED_64) {
        pinned_mr = PinnedMemoryResource::make_fixed_sized_if_available(
            get_current_numa_node(), pool_properties, /*block_size=*/64
        );
    } else if (kind == BufferKind::PINNED_128) {
        pinned_mr = PinnedMemoryResource::make_fixed_sized_if_available(
            get_current_numa_node(), pool_properties, /*block_size=*/128
        );
    }
    return std::make_shared<BufferResource>(
        cudf::get_current_device_resource_ref(),
        std::move(pinned_mr),
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{},
        std::nullopt,
        std::move(pool)
    );
}

}  // namespace

/**
 * @brief Parameterized test fixture for `Buffer::copy_to`.
 *
 * Each `CopyToParam` specifies:
 *   - src_kind / dst_kind — memory kind of the source and destination buffers
 *   - copy_size           — bytes to copy (0, 11, 64, 128, 256)
 *   - src_offset          — byte offset into the source buffer (0 or 512)
 *   - dst_offset          — byte offset into the destination buffer (0 or 512)
 *
 * Both buffers are 1 KiB.  All (copy_size, offset) pairs satisfy
 * `copy_size + offset ≤ 1024`, so every combination is in-bounds.
 *
 * Two independent BufferResources are created — one for the source and one for
 * the destination — so that PINNED_64 and PINNED_128 can coexist in the same
 * test case (each BR holds its own PinnedMemoryResource with a distinct block size).
 */
class BufferCopyToTest : public ::testing::TestWithParam<CopyToParam> {
  protected:
    static constexpr std::size_t kBufferSize = 1024;  // 1 KiB

    void SetUp() override {
        auto const& p = GetParam();

        if ((kind_needs_pinned(p.src_kind) || kind_needs_pinned(p.dst_kind))
            && !is_pinned_memory_resources_supported())
        {
            GTEST_SKIP() << "Pinned memory resources are not supported on this system";
        }

        stream_pool = std::make_shared<rmm::cuda_stream_pool>(2);
        src_br = make_copy_test_br(p.src_kind, stream_pool);
        dst_br = make_copy_test_br(p.dst_kind, stream_pool);
    }

    /// Read back @p size bytes from @p buf starting at @p offset into a vector.
    /// Uses exclusive_data_access_blocks() so it works for all storage types.
    std::vector<std::uint8_t> ReadBackFromBuffer(
        Buffer& buf, std::size_t size, std::size_t offset
    ) {
        std::vector<std::uint8_t> result(size);
        auto blocks = buf.exclusive_data_access_blocks();
        std::size_t const block_size = kBufferSize / blocks.size();
        std::size_t flat_off = offset;
        std::size_t result_off = 0;
        std::size_t bytes_left = size;
        while (bytes_left > 0) {
            std::size_t const bi = flat_off / block_size;
            std::size_t const off = flat_off % block_size;
            std::size_t const n = std::min(bytes_left, block_size - off);
            RAPIDSMPF_CUDA_TRY(cudaMemcpy(
                result.data() + result_off, blocks[bi] + off, n, cudaMemcpyDefault
            ));
            flat_off += n;
            result_off += n;
            bytes_left -= n;
        }
        buf.unlock();
        return result;
    }

    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    std::shared_ptr<BufferResource> src_br;
    std::shared_ptr<BufferResource> dst_br;
};

TEST_P(BufferCopyToTest, CopiesDataCorrectly) {
    auto const& p = GetParam();
    MemoryType const src_type = to_memory_type(p.src_kind);
    MemoryType const dst_type = to_memory_type(p.dst_kind);

    // A single shared stream keeps all operations sequentially ordered, which
    // simplifies synchronization: after one stream.synchronize() every prior
    // operation on that stream is complete.
    auto stream = stream_pool->get_stream();

    // Source pattern: byte i == uint8_t(i), wrapping at 256.
    auto const monotonic = iota_vector<uint8_t>(kBufferSize);

    // ---- Allocate and initialize the source buffer ----

    auto [src_alloc, src_ob] =
        src_br->reserve(src_type, kBufferSize, AllowOverbooking::YES);
    auto src_buf = src_br->allocate(kBufferSize, stream, src_alloc);

    std::size_t src_offset = 0;
    src_buf->write_access_blocks([&](std::span<std::byte> block,
                                     rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            block.data(),
            monotonic.data() + src_offset,
            block.size(),
            cudaMemcpyDefault,
            stream
        ));
        src_offset += block.size();
    });

    // ---- Allocate the destination buffer (leave uninitialized) ----

    auto [dst_alloc, dst_ob] =
        dst_br->reserve(dst_type, kBufferSize, AllowOverbooking::YES);
    auto dst_buf = dst_br->allocate(kBufferSize, stream, dst_alloc);

    // ---- The operation under test: src -> dst ----

    src_buf->copy_to(*dst_buf, p.copy_size, p.dst_offset, p.src_offset);

    // copy_to enqueues on dst stream; wait for completion.
    stream.synchronize();

    if (p.copy_size == 0) {
        return;  // Zero-size copy: verify only that no exception was thrown.
    }

    auto to_string = [](auto const& vec, size_t offset, size_t size) {
        std::stringstream ss;
        for (size_t i = 0; i < size; ++i) {
            ss << static_cast<int>(vec.at(offset + i)) << " ";
        }
        return ss.str();
    };

    SCOPED_TRACE("src: " + to_string(monotonic, p.src_offset, p.copy_size));

    // ---- Read back from dst and verify ----
    {
        auto dst_result = ReadBackFromBuffer(
            *dst_buf, p.copy_size, static_cast<std::size_t>(p.dst_offset)
        );
        SCOPED_TRACE("dst: " + to_string(dst_result, 0, dst_result.size()));
        EXPECT_TRUE(
            std::equal(
                monotonic.begin() + p.src_offset,
                monotonic.begin() + p.src_offset + p.copy_size,
                dst_result.begin()
            )
        );
    }
}

/// @brief Generate all (src_kind × dst_kind × copy_size × src_offset × dst_offset)
/// combinations.
std::vector<CopyToParam> all_copy_to_params() {
    constexpr std::array kinds{
        BufferKind::DEVICE,
        BufferKind::HOST,
        BufferKind::PINNED_64,
        BufferKind::PINNED_128
    };
    constexpr std::array copy_sizes{0, 11, 64, 128, 256};
    constexpr std::array src_offsets{0, 111, 512};
    constexpr std::array dst_offsets{0, 111, 512};

    std::vector<CopyToParam> params;
    for (auto src : kinds) {
        for (auto dst : kinds) {
            for (std::size_t sz : copy_sizes) {
                for (std::ptrdiff_t src_off : src_offsets) {
                    for (std::ptrdiff_t dst_off : dst_offsets) {
                        params.push_back({src, dst, sz, src_off, dst_off});
                    }
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AllPairs,
    BufferCopyToTest,
    ::testing::ValuesIn(all_copy_to_params()),
    [](::testing::TestParamInfo<CopyToParam> const& info) {
        auto const& p = info.param;
        return std::string(buffer_kind_to_string(p.src_kind)) + "_to_"
               + std::string(buffer_kind_to_string(p.dst_kind)) + "_size"
               + std::to_string(p.copy_size) + "_srcoff" + std::to_string(p.src_offset)
               + "_dstoff" + std::to_string(p.dst_offset);
    }
);
