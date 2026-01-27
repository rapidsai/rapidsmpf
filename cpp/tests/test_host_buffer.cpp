/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <rapidsmpf/utils/misc.hpp>

#include "utils.hpp"

class HostMemoryResource : public ::testing::TestWithParam<size_t> {
  protected:
    void SetUp() override {
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            p_mr = std::make_shared<rapidsmpf::HostMemoryResource>();
        } else {
            GTEST_SKIP() << "HostBuffer is not supported for CUDA versions "
                            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR;
        }
    }

    void TearDown() override {
        p_mr.reset();
    }

    void test_buffer(auto&& buffer, std::vector<uint8_t> const& source_data) {
        ASSERT_EQ(source_data.size(), buffer.size());
        ASSERT_NE(nullptr, buffer.data());

        // Synchronize on stream to ensure copy is complete
        buffer.stream().synchronize();

        const auto* data = buffer.data();
        // Check the contents using std::equal
        EXPECT_TRUE(
            std::equal(
                source_data.begin(),
                source_data.end(),
                reinterpret_cast<const uint8_t*>(data)
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

    rmm::cuda_stream_view stream{};
    std::shared_ptr<rapidsmpf::HostMemoryResource> p_mr;
    rmm::mr::cuda_async_memory_resource cuda_mr{};
};

// Test with various buffer sizes
INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    HostMemoryResource,
    ::testing::Values(
        1,  // 1B
        9,  // 9B
        1024,  // 1KB
        1048576  // 1MB
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(HostMemoryResource, from_owned_vector) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create a host buffer by taking ownership of a vector
    auto buffer = rapidsmpf::HostBuffer::from_owned_vector(
        std::vector<uint8_t>(source_data), stream, *p_mr
    );

    EXPECT_NO_THROW(test_buffer(std::move(buffer), source_data));
}

TEST_P(HostMemoryResource, from_uint8_vector) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create a host buffer by copying the vector
    auto buffer = rapidsmpf::HostBuffer::from_uint8_vector(source_data, stream, *p_mr);

    EXPECT_NO_THROW(test_buffer(std::move(buffer), source_data));
}

class PinnedResource : public HostMemoryResource {
  protected:
    void SetUp() override {
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            p_mr = std::make_shared<rapidsmpf::PinnedMemoryResource>();
        } else {
            GTEST_SKIP() << "HostBuffer is not supported for CUDA versions "
                            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR;
        }
    }
};

// Test with various buffer sizes
INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    PinnedResource,
    ::testing::Values(
        1,  // 1B
        9,  // 9B
        1024,  // 1KB
        1048576  // 1MB
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(PinnedResource, from_uint8_vector) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create a host buffer by copying the vector
    auto buffer = rapidsmpf::HostBuffer::from_uint8_vector(source_data, stream, *p_mr);

    EXPECT_NO_THROW(test_buffer(std::move(buffer), source_data));
}

TEST_P(PinnedResource, from_owned_vector) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create a host buffer by taking ownership of a vector
    auto buffer = rapidsmpf::HostBuffer::from_owned_vector(
        std::vector<uint8_t>(source_data), stream, *p_mr
    );

    EXPECT_NO_THROW(test_buffer(std::move(buffer), source_data));
}

TEST_P(PinnedResource, from_rmm_device_buffer) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    auto& pinned_mr = dynamic_cast<rapidsmpf::PinnedMemoryResource&>(*p_mr);

    // create a pinned host buffer using pinned_mr
    auto pinned_host_buffer = std::make_unique<rmm::device_buffer>(
        source_data.data(), source_data.size(), stream, pinned_mr
    );

    // Create a host buffer by taking ownership of an rmm::device_buffer
    auto buffer = rapidsmpf::HostBuffer::from_rmm_device_buffer(
        std::move(pinned_host_buffer), stream, pinned_mr
    );

    EXPECT_NO_THROW(test_buffer(std::move(buffer), source_data));
}
