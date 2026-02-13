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

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
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
        EXPECT_TRUE(std::equal(
            source_data.begin(), source_data.end(), reinterpret_cast<const uint8_t*>(data)
        ));

        // move constructor
        rapidsmpf::HostBuffer buffer2(std::move(buffer));
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

// -----------------------------------------------------------------------------
// FixedSizedHostBuffer tests (vector-based factories only)
// -----------------------------------------------------------------------------

class FixedSizedHostBufferTest : public ::testing::Test {};

TEST_F(FixedSizedHostBufferTest, DefaultConstructedIsEmpty) {
    rapidsmpf::FixedSizedHostBuffer buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.total_size(), 0u);
    EXPECT_EQ(buf.block_size(), 0u);
    EXPECT_EQ(buf.num_blocks(), 0u);
    EXPECT_TRUE(buf.blocks().empty());
}

TEST_F(FixedSizedHostBufferTest, FromVectorOneBlock) {
    auto buf =
        rapidsmpf::FixedSizedHostBuffer::from_vector(std::vector<std::byte>{100}, 64);
    EXPECT_EQ(buf.total_size(), 1);
    EXPECT_EQ(buf.num_blocks(), 1);
    EXPECT_EQ(buf.block_size(), 64);
}

TEST_F(FixedSizedHostBufferTest, FromVectorSingleBlock) {
    std::vector<std::byte> vec(100);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<std::byte>(i & 0xFF);
    }
    auto buf = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec), 100);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.total_size(), 100u);
    EXPECT_EQ(buf.block_size(), 100u);
    EXPECT_EQ(buf.num_blocks(), 1u);
    ASSERT_EQ(buf.blocks().size(), 1u);
    auto block = buf.block_data(0);
    EXPECT_EQ(block.size(), 100u);
}

// TEST_F(FixedSizedHostBufferTest, FromVectorMultipleBlocks) {
//     std::vector<std::byte> vec(256);
//     for (std::size_t i = 0; i < vec.size(); ++i) {
//         vec[i] = static_cast<std::byte>(i & 0xFF);
//     }
//     const std::size_t block_size = 64;
//     auto buf = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec),
//     block_size); EXPECT_FALSE(buf.empty()); EXPECT_EQ(buf.total_size(), 256u);
//     EXPECT_EQ(buf.block_size(), block_size);
//     EXPECT_EQ(buf.num_blocks(), 4u);
//     ASSERT_EQ(buf.blocks().size(), 4u);
//     for (std::size_t b = 0; b < buf.num_blocks(); ++b) {
//         auto block = buf.block_data(b);
//         EXPECT_EQ(block.size(), block_size);
//         auto const base = b * block_size;
//         auto expected = std::views::iota(base, base + block_size)
//                         | std::views::transform([](std::size_t i) {
//                               return static_cast<std::byte>(i & 0xFF);
//                           });
//         EXPECT_TRUE(std::ranges::equal(block, expected));
//     }
// }

// TEST_F(FixedSizedHostBufferTest, FromVectorBlockDataOutOfRangeThrows) {
//     std::vector<std::byte> vec(64);
//     auto buf = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec), 64);
//     EXPECT_THROW(static_cast<void>(buf.block_data(1)), std::out_of_range);
// }

// TEST_F(FixedSizedHostBufferTest, FromVectorsEmpty) {
//     auto buf =
//         rapidsmpf::FixedSizedHostBuffer::from_vectors(std::vector<std::vector<std::byte>>{
//         });
//     EXPECT_TRUE(buf.empty());
//     EXPECT_EQ(buf.total_size(), 0u);
//     EXPECT_EQ(buf.num_blocks(), 0u);
// }

// TEST_F(FixedSizedHostBufferTest, FromVectorsMultipleBlocks) {
//     const std::size_t block_sz = 32;
//     const std::size_t n_blocks = 4;
//     std::vector<std::vector<std::byte>> vecs(n_blocks);
//     for (std::size_t b = 0; b < n_blocks; ++b) {
//         vecs[b].resize(block_sz);
//         for (std::size_t i = 0; i < block_sz; ++i) {
//             vecs[b][i] = static_cast<std::byte>((b * block_sz + i) & 0xFF);
//         }
//     }
//     auto buf = rapidsmpf::FixedSizedHostBuffer::from_vectors(std::move(vecs));
//     EXPECT_FALSE(buf.empty());
//     EXPECT_EQ(buf.total_size(), n_blocks * block_sz);
//     EXPECT_EQ(buf.block_size(), block_sz);
//     EXPECT_EQ(buf.num_blocks(), n_blocks);
//     for (std::size_t b = 0; b < buf.num_blocks(); ++b) {
//         auto block = buf.block_data(b);
//         EXPECT_EQ(block.size(), block_sz);
//         auto const base = b * block_sz;
//         auto expected = std::views::iota(base, base + block_sz)
//                         | std::views::transform([](std::size_t i) {
//                               return static_cast<std::byte>(i & 0xFF);
//                           });
//         EXPECT_TRUE(std::ranges::equal(block, expected));
//     }
// }

// TEST_F(FixedSizedHostBufferTest, Reset) {
//     std::vector<std::byte> vec(64);
//     auto buf = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec), 64);
//     EXPECT_FALSE(buf.empty());
//     buf.reset();
//     EXPECT_TRUE(buf.empty());
//     EXPECT_EQ(buf.total_size(), 0u);
//     EXPECT_EQ(buf.block_size(), 0u);
//     EXPECT_EQ(buf.num_blocks(), 0u);
//     EXPECT_TRUE(buf.blocks().empty());
// }

// TEST_F(FixedSizedHostBufferTest, MoveConstructor) {
//     std::vector<std::byte> vec(128);
//     auto buf1 = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec), 64);
//     auto buf2 = rapidsmpf::FixedSizedHostBuffer(std::move(buf1));
//     EXPECT_TRUE(buf1.empty());
//     EXPECT_EQ(buf1.num_blocks(), 0u);
//     EXPECT_FALSE(buf2.empty());
//     EXPECT_EQ(buf2.total_size(), 128u);
//     EXPECT_EQ(buf2.num_blocks(), 2u);
// }

// TEST_F(FixedSizedHostBufferTest, MoveAssignment) {
//     std::vector<std::byte> vec(64);
//     auto buf1 = rapidsmpf::FixedSizedHostBuffer::from_vector(std::move(vec), 64);
//     rapidsmpf::FixedSizedHostBuffer buf2;
//     buf2 = std::move(buf1);
//     EXPECT_TRUE(buf1.empty());
//     EXPECT_EQ(buf1.num_blocks(), 0u);
//     EXPECT_FALSE(buf2.empty());
//     EXPECT_EQ(buf2.total_size(), 64u);
//     EXPECT_EQ(buf2.num_blocks(), 1u);
// }
