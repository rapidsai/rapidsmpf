/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>

#include "utils.hpp"

class PinnedHostBufferTest : public ::testing::TestWithParam<size_t> {
  protected:
    void SetUp() override {
        p_pool = std::make_unique<rapidsmpf::PinnedMemoryPool>(0);
        p_mr = std::make_shared<rapidsmpf::PinnedMemoryResource>(*p_pool);
    }

    void TearDown() override {
        p_mr.reset();
        p_pool.reset();
    }

    cuda::stream_ref stream{};
    std::unique_ptr<rapidsmpf::PinnedMemoryPool> p_pool;
    std::shared_ptr<rapidsmpf::PinnedMemoryResource> p_mr;

    rmm::mr::cuda_async_memory_resource cuda_mr{};
};

TEST_P(PinnedHostBufferTest, synchronized_host_data) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create pinned buffer using deep copy constructor
    rapidsmpf::PinnedHostBuffer buffer(source_data.data(), buffer_size, stream, p_mr);

    // Synchronize on stream to ensure copy is complete
    buffer.synchronize();

    // Check the contents using std::equal
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    const auto* data = buffer.data();
    EXPECT_TRUE(std::equal(
        source_data.begin(), source_data.end(), reinterpret_cast<const uint8_t*>(data)
    ));

    // move constructor
    rapidsmpf::PinnedHostBuffer buffer2(std::move(buffer));
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(std::equal(
        source_data.begin(),
        source_data.end(),
        reinterpret_cast<const uint8_t*>(buffer2.data())
    ));
    EXPECT_EQ(data, buffer2.data());

    // move assignment
    buffer = std::move(buffer2);
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(std::equal(
        source_data.begin(),
        source_data.end(),
        reinterpret_cast<const uint8_t*>(buffer.data())
    ));
    EXPECT_EQ(data, buffer.data());

    // deep copy
    rapidsmpf::PinnedHostBuffer buffer3(buffer, stream, p_mr);
    buffer3.synchronize();
    EXPECT_TRUE(std::equal(
        source_data.begin(),
        source_data.end(),
        reinterpret_cast<const uint8_t*>(buffer3.data())
    ));

    // Clean up
    buffer.deallocate_async();
    buffer3.deallocate_async();
}

TEST_P(PinnedHostBufferTest, device_data) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto host_data = random_vector<uint8_t>(0, buffer_size);
    rmm::device_buffer dev_data(host_data.data(), buffer_size, stream, cuda_mr);

    // Create pinned buffer by copying device data on the same stream
    rapidsmpf::PinnedHostBuffer buffer(dev_data.data(), buffer_size, stream, p_mr);

    // Check the contents using std::equal
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    buffer.synchronize();
    EXPECT_TRUE(std::equal(
        host_data.begin(),
        host_data.end(),
        reinterpret_cast<const uint8_t*>(buffer.data())
    ));
}

// Test with various buffer sizes
INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    PinnedHostBufferTest,
    ::testing::Values(
        1024,  // 1KB
        4096,  // 4KB
        16384,  // 16KB
        65536,  // 64KB
        262144,  // 256KB
        1048576,  // 1MB
        4194304  // 4MB
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);
