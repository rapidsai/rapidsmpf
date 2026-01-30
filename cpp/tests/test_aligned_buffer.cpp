/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <rapidsmpf/memory/detail/aligned_buffer.hpp>

namespace {

class AlignedBufferTest : public ::testing::TestWithParam<std::size_t> {
  protected:
    void run_alignment_check(std::size_t alignment) {
        constexpr std::size_t size = 1 << 20;

        rapidsmpf::detail::AlignedBuffer buffer(stream, mr, size, alignment);

        ASSERT_NE(buffer.data(), nullptr);
        EXPECT_EQ(buffer.size(), size);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % alignment, 0u);

        rapidsmpf::detail::AlignedBuffer moved(std::move(buffer));
        EXPECT_EQ(buffer.data(), nullptr);
        ASSERT_NE(moved.data(), nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(moved.data()) % alignment, 0u);
    }

    rmm::cuda_stream_view stream{};
    rmm::mr::cuda_async_memory_resource mr{};
};

INSTANTIATE_TEST_SUITE_P(
    SupportedAlignments,
    AlignedBufferTest,
    ::testing::Values(64u, 128u, 256u),
    [](const ::testing::TestParamInfo<std::size_t>& info) {
        return std::to_string(info.param);
    }
);

INSTANTIATE_TEST_SUITE_P(
    UnsupportedAlignments,
    AlignedBufferTest,
    ::testing::Values(512u, 1024u, 4096u, 8192u),
    [](const ::testing::TestParamInfo<std::size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(AlignedBufferTest, AlignmentBehavior) {
    const std::size_t alignment = GetParam();

    if (alignment <= 256u) {
        EXPECT_NO_THROW(run_alignment_check(alignment));
    } else {
        EXPECT_THROW(run_alignment_check(alignment), rmm::bad_alloc);
    }
}

}  // namespace
