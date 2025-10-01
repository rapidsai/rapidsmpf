/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>

class PinnedHostBufferTest : public ::testing::TestWithParam<size_t> {
  protected:
    void SetUp() override {
        stream = rmm::cuda_stream_default;
        p_pool = std::make_unique<rapidsmpf::PinnedMemoryPool>(0);
        p_resource = std::make_unique<rapidsmpf::PinnedMemoryResource>(*p_pool);
    }

    void TearDown() override {
        p_resource.reset();
        p_pool.reset();
    }

    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::PinnedMemoryPool> p_pool;
    std::unique_ptr<rapidsmpf::PinnedMemoryResource> p_resource;
};

TEST_P(PinnedHostBufferTest, BufferWithDeepCopy) {
    const size_t buffer_size = GetParam();
    const size_t num_elements = buffer_size / sizeof(int);

    // Create a vector with random data
    std::vector<int> source_data(num_elements);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 1000);

    for (auto& val : source_data) {
        val = dis(gen);
    }

    // Create pinned buffer using deep copy constructor
    rapidsmpf::PinnedHostBuffer buffer(
        source_data.data(), buffer_size, stream, p_resource.get()
    );

    // Synchronize on stream to ensure copy is complete
    stream.synchronize();

    // Check the contents using std::memcmp
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    EXPECT_TRUE(
        std::equal(
            source_data.begin(), source_data.end(), static_cast<int*>(buffer.data())
        )
    );

    // Clean up
    buffer.deallocate_async();
    stream.synchronize();
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
