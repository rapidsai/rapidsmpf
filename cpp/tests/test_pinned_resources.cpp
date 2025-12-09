/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils.hpp>

#include "utils.hpp"

class PinnedResource : public ::testing::TestWithParam<size_t> {
  protected:
    void SetUp() override {
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            p_mr = std::make_shared<rapidsmpf::PinnedMemoryResource>();
        } else {
            GTEST_SKIP() << "HostBuffer is not supported for CUDA versions "
                            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR;
        }
    }

    void TearDown() override {
        p_mr.reset();
    }

    rmm::cuda_stream_view stream{};
    std::shared_ptr<rapidsmpf::PinnedMemoryResource> p_mr;
    rmm::mr::cuda_async_memory_resource cuda_mr{};
};

// Test with various buffer sizes
INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    PinnedResource,
    ::testing::Values(
        1,  // 1B
        1024,  // 1KB
        1048576  // 1MB
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(PinnedResource, HostBuffer) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create pinned buffer using deep copy constructor
    auto buffer = rapidsmpf::HostBuffer::from_uint8_vector(source_data, stream, *p_mr);

    // Synchronize on stream to ensure copy is complete
    buffer.stream().synchronize();

    // Check the contents using std::equal
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    const auto* data = buffer.data();
    EXPECT_TRUE(
        std::equal(
            source_data.begin(), source_data.end(), reinterpret_cast<const uint8_t*>(data)
        )
    );

    // move constructor
    rapidsmpf::HostBuffer buffer2(std::move(buffer));
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(
        std::equal(
            source_data.begin(),
            source_data.end(),
            reinterpret_cast<const uint8_t*>(buffer2.data())
        )
    );
    EXPECT_EQ(data, buffer2.data());

    // move assignment
    buffer = std::move(buffer2);
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(
        std::equal(
            source_data.begin(),
            source_data.end(),
            reinterpret_cast<const uint8_t*>(buffer.data())
        )
    );
    EXPECT_EQ(data, buffer.data());

    // Clean up
    buffer.deallocate_async();
    buffer2.deallocate_async();
}
