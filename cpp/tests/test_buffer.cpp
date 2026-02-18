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
            PinnedMemoryResource::make_if_available(),
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

    buffer_copy(*buffer2, *buffer1, buffer_size);

    std::vector<std::uint8_t> result(buffer_size);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        result.data(), buffer2->data(), buffer_size, cudaMemcpyDefault, stream2
    ));
    stream2.synchronize();

    EXPECT_EQ(result, random_data);
}

TEST_P(BufferRebindStreamTest, RebindStreamSynchronizesCorrectly) {
    MemoryType mem_type = GetParam();
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
