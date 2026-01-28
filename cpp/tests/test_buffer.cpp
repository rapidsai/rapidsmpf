/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

class BufferRebindStreamTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stream_pool = std::make_shared<rmm::cuda_stream_pool>(2);
        br = std::make_unique<BufferResource>(
            cudf::get_current_device_resource_ref(),
            PinnedMemoryResource::Disabled,
            std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{},
            std::nullopt,
            stream_pool
        );

        std::mt19937 rng(42);
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        random_data.resize(buffer_size);
        for (auto& byte : random_data) {
            byte = dist(rng);
        }
    }

    static constexpr std::size_t buffer_size = 32_MiB;
    static constexpr std::size_t chunk_size = 1_MiB;

    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    std::unique_ptr<BufferResource> br;
    std::vector<uint8_t> random_data;
};

// test rebinding stream and copy data between buffers on different streams
TEST_F(BufferRebindStreamTest, RebindStreamAndCopy) {
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    auto rmm_buffer = std::make_unique<rmm::device_buffer>(
        random_data.data(), buffer_size, stream1, br->device_mr()
    );

    auto [reserve1, overbooking1] =
        br->reserve(MemoryType::DEVICE, buffer_size, AllowOverbooking::YES);
    auto device_buffer1 = br->allocate(buffer_size, stream1, reserve1);
    EXPECT_EQ(device_buffer1->stream().value(), stream1.value());

    std::size_t num_chunks = buffer_size / chunk_size;
    // copy 1MB parts iteratively from rmm_buffer to device_buffer1 on stream1
    for (std::size_t i = 0; i < num_chunks; ++i) {
        std::size_t offset = i * chunk_size;
        device_buffer1->write_access([&](std::byte* dst, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                dst + offset,
                static_cast<std::byte*>(rmm_buffer->data()) + offset,
                chunk_size,
                cudaMemcpyDefault,
                stream
            ));
        });
    }

    // allocate buffer2 on stream2
    auto [reserve2, overbooking2] =
        br->reserve(MemoryType::DEVICE, buffer_size, AllowOverbooking::YES);
    auto device_buffer2 = br->allocate(buffer_size, stream2, reserve2);
    EXPECT_EQ(device_buffer2->stream().value(), stream2.value());

    // rebind buffer1 to stream2
    device_buffer1->rebind_stream(stream2);
    EXPECT_EQ(device_buffer1->stream().value(), stream2.value());

    // copy buffer1 to buffer2 on stream2
    buffer_copy(*device_buffer2, *device_buffer1, buffer_size);

    std::vector<uint8_t> result(buffer_size);  // copy to host and verify
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        result.data(), device_buffer2->data(), buffer_size, cudaMemcpyDefault, stream2
    ));
    stream2.synchronize();

    EXPECT_EQ(result, random_data);
}

// test rebinding stream and access the same buffer on different streams
TEST_F(BufferRebindStreamTest, RebindStreamSynchronizesCorrectly) {
    auto stream1 = stream_pool->get_stream();
    auto stream2 = stream_pool->get_stream();
    ASSERT_NE(stream1.value(), stream2.value());

    constexpr std::size_t test_size = 4_MiB;

    auto [reserve1, overbooking1] =
        br->reserve(MemoryType::DEVICE, test_size, AllowOverbooking::YES);
    auto buffer1 = br->allocate(test_size, stream1, reserve1);

    buffer1->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0xAB, test_size, stream));
    });

    buffer1->rebind_stream(stream2);
    EXPECT_EQ(buffer1->stream().value(), stream2.value());

    buffer1->write_access([&](std::byte* ptr, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0xCD, test_size / 2, stream));
    });

    std::vector<uint8_t> result(test_size);  // copy to host and verify
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
