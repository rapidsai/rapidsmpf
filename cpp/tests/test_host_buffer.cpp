/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/fixed_sized_host_buffer.hpp>
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

// Test for various vector sizes with a fixed block size
class FixedSizedHostBufferTest : public ::testing::TestWithParam<size_t> {
  public:
    static constexpr size_t block_size = 32;
};

INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    FixedSizedHostBufferTest,
    ::testing::Values(0, 1, 10, FixedSizedHostBufferTest::block_size, 1000),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(FixedSizedHostBufferTest, from_vector) {
    auto source_data = iota_vector<std::byte>(GetParam());
    auto const expected = source_data;

    auto check_buf = [&](auto const& buf) {
        EXPECT_EQ(expected.size(), buf.total_size());
        EXPECT_EQ(block_size, buf.block_size());
        EXPECT_EQ((expected.size() + block_size - 1) / block_size, buf.num_blocks());
        for (size_t i = 0; i < buf.num_blocks(); ++i) {
            EXPECT_EQ(block_size, buf.block_data(i).size());
            size_t offset = i * block_size;
            EXPECT_TRUE(
                std::equal(
                    expected.begin() + offset,
                    expected.begin() + std::min(offset + block_size, expected.size()),
                    buf.block_data(i).data()
                )
            );
        }
    };

    auto buf0 =
        rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(source_data), block_size);
    check_buf(buf0);

    rapidsmpf::FixedSizedHostBuffer buf1(std::move(buf0));
    EXPECT_TRUE(buf0.empty());
    check_buf(buf1);

    buf0 = std::move(buf1);
    EXPECT_TRUE(buf1.empty());
    check_buf(buf0);
}

TEST_P(FixedSizedHostBufferTest, from_vectors) {
    size_t const num_vectors = GetParam();

    std::vector<std::vector<std::byte>> vecs;
    vecs.reserve(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vecs.emplace_back(
            iota_vector<std::byte>(
                block_size, static_cast<std::byte>(i * block_size & 0xff)
            )
        );
    }
    auto const expected_vecs = vecs;

    auto check_buf = [&](auto const& buf) {
        EXPECT_EQ(num_vectors * block_size, buf.total_size());
        EXPECT_EQ(num_vectors > 0 ? block_size : 0, buf.block_size());
        EXPECT_EQ(num_vectors, buf.num_blocks());
        for (size_t i = 0; i < buf.num_blocks(); ++i) {
            EXPECT_EQ(block_size, buf.block_data(i).size());
            EXPECT_TRUE(
                std::equal(
                    expected_vecs[i].begin(),
                    expected_vecs[i].end(),
                    buf.block_data(i).data()
                )
            );
        }
    };

    auto buf0 = rapidsmpf::FixedSizedHostBuffer::from_vectors(std::move(vecs));
    check_buf(buf0);

    rapidsmpf::FixedSizedHostBuffer buf1(std::move(buf0));
    EXPECT_TRUE(buf0.empty());
    check_buf(buf1);

    buf0 = std::move(buf1);
    EXPECT_TRUE(buf1.empty());
    check_buf(buf0);
}

TEST_P(FixedSizedHostBufferTest, from_multi_blocks_alloc) {
    size_t const num_buffers = GetParam();

    rmm::mr::pinned_host_memory_resource upstream_mr;
    constexpr std::size_t mem_limit = 4 * 1024 * 1024;
    constexpr std::size_t capacity = 4 * 1024 * 1024;
    cucascade::memory::fixed_size_host_memory_resource host_mr(
        0, upstream_mr, mem_limit, capacity, block_size
    );

    std::size_t const allocation_size = num_buffers * block_size;
    auto allocation = host_mr.allocate_multiple_blocks(allocation_size);

    std::vector<std::vector<std::byte>> vecs;
    for (size_t i = 0; i < allocation->size(); ++i) {
        auto block = (*allocation)[i];
        auto& fill = vecs.emplace_back(
            iota_vector<std::byte>(
                block_size, static_cast<std::byte>(i * block_size & 0xff)
            )
        );
        std::ranges::copy(fill, block.begin());
    }

    auto check_buf = [&](auto const& buf) {
        EXPECT_EQ(num_buffers * block_size, buf.total_size());
        EXPECT_EQ(num_buffers > 0 ? block_size : 0, buf.block_size());
        EXPECT_EQ(num_buffers, buf.num_blocks());
        for (size_t i = 0; i < buf.num_blocks(); ++i) {
            EXPECT_EQ(block_size, buf.block_data(i).size());
            EXPECT_TRUE(std::ranges::equal(vecs[i], buf.block_data(i)));
        }
    };

    auto buf0 =
        rapidsmpf::FixedSizedHostBuffer::from_multi_blocks_alloc(std::move(allocation));
    check_buf(buf0);

    rapidsmpf::FixedSizedHostBuffer buf1(std::move(buf0));
    EXPECT_TRUE(buf0.empty());
    check_buf(buf1);

    buf0 = std::move(buf1);
    EXPECT_TRUE(buf1.empty());
    check_buf(buf0);
}

TEST(FixedSizedHostBufferTest, empty_equality) {
    std::array bufs{
        rapidsmpf::FixedSizedHostBuffer{},
        rapidsmpf::FixedSizedHostBuffer::from_vector({}, 10),
        rapidsmpf::FixedSizedHostBuffer::from_vectors({}),
        rapidsmpf::FixedSizedHostBuffer::from_multi_blocks_alloc({})
    };

    for (size_t i = 0; i < bufs.size(); ++i) {
        for (size_t j = i; j < bufs.size(); ++j) {
            EXPECT_EQ(bufs[i], bufs[j]);
        }
    }
}
