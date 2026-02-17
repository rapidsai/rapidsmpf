/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
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

namespace {

/// Discover the actual pool size the driver creates when a small max is requested.
/// Creates a pool with \p requested_max_pool_size (e.g. 1 MiB), then uses recursive
/// doubling of allocation size until allocation fails; returns the last successful size.
std::size_t discover_pinned_pool_actual_size(
    rmm::cuda_stream_view stream, std::size_t requested_max_pool_size = 1_MiB
) {
    rapidsmpf::PinnedMemoryResource pinned_mr{
        rapidsmpf::get_current_numa_node(),
        rapidsmpf::PinnedPoolProperties{.max_pool_size = requested_max_pool_size}
    };

    auto can_allocate = [&](size_t size) -> bool {
        try {
            void* ptr = pinned_mr.allocate(stream, size);
            pinned_mr.deallocate(stream, ptr, size);
            return true;
        } catch (cuda::cuda_error const&) {
            return false;
        }
    };

    constexpr std::size_t alignment = cuda::mr::default_cuda_malloc_alignment;

    // Advance max size until we can't allocate using recursive doubling (guard overflow).
    std::size_t max_size = requested_max_pool_size;
    while (can_allocate(max_size)
           && max_size <= std::numeric_limits<std::size_t>::max() / 2)
    {
        max_size *= 2;
    }
    max_size = std::max(max_size / 2, requested_max_pool_size);

    // Bisection search for the actual pool size; min_size is a known-good lower bound.
    std::size_t min_size = std::max(max_size / 2, requested_max_pool_size);
    while (min_size + alignment <= max_size) {
        std::size_t mid_size = std::midpoint(min_size, max_size);
        mid_size = ((mid_size + alignment - 1) / alignment) * alignment;
        mid_size = std::min(mid_size, max_size);  // clamp after rounding
        if (can_allocate(mid_size)) {
            min_size = mid_size;
        } else {
            max_size = mid_size - alignment;
        }
    }
    return min_size;
}

}  // namespace

TEST(PinnedResourceMaxSize, max_pool_size_limit) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "PinnedMemoryResource is not supported";
    }

    // Ensure a current device context so driver APIs
    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));
    auto stream = cudf::get_default_stream();

    // Create a PinnedMemoryResource with max pool size of 1 MiB; driver may round up.
    rapidsmpf::PinnedMemoryResource pinned_mr{
        rapidsmpf::get_current_numa_node(),
        rapidsmpf::PinnedPoolProperties{.initial_pool_size = 0, .max_pool_size = 1_MiB}
    };

    auto alloc_and_dealloc = [&](std::size_t size) {
        void* ptr = pinned_mr.allocate(stream, size);
        EXPECT_NE(nullptr, ptr);
        pinned_mr.deallocate(stream, ptr, size);
    };

    alloc_and_dealloc(512_KiB);

    // Find the actual pool size (driver may round up, e.g. to 32 MiB) experimentally.
    std::size_t const actual_pool_size = discover_pinned_pool_actual_size(stream, 1_MiB);
    EXPECT_THROW(alloc_and_dealloc(actual_pool_size + 1), cuda::cuda_error);
    stream.synchronize();
}
