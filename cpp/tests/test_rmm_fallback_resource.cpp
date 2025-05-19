/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cstddef>
#include <stdexcept>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/buffer/rmm_fallback_resource.hpp>


using namespace rapidsmpf;

template <typename ExceptionType>
struct throw_at_limit_resource final : public rmm::mr::device_memory_resource {
    throw_at_limit_resource(std::size_t limit) : limit{limit} {}

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        if (bytes > limit) {
            throw ExceptionType{"foo"};
        }
        void* ptr{nullptr};
        RMM_CUDA_TRY_ALLOC(cudaMallocAsync(&ptr, bytes, stream));
        allocs.insert(ptr);
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override {
        RMM_ASSERT_CUDA_SUCCESS(cudaFree(ptr));
        allocs.erase(ptr);
    }

    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other
    ) const noexcept override {
        return this == &other;
    }

    const std::size_t limit;
    std::unordered_set<void*> allocs{};
};

TEST(FailureAlternateTest, TrackBothUpstreams) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{100};
    throw_at_limit_resource<rmm::out_of_memory> alternate_mr{1000};
    RmmFallbackResource<rmm::out_of_memory> mr{primary_mr, alternate_mr};

    // Check that a small allocation goes to the primary resource
    {
        void* a1 = mr.allocate(10);
        EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{a1});
        EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
        mr.deallocate(a1, 10);
        EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
        EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
    }

    // Check that a large allocation goes to the alternate resource
    {
        void* a1 = mr.allocate(200);
        EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
        EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{a1});
        mr.deallocate(a1, 200);
        EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
        EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
    }

    // Check that the exceptions raised by the alternate isn't caught
    EXPECT_THROW(mr.allocate(2000), rmm::out_of_memory);
}

TEST(FailureAlternateTest, DifferentExceptionTypes) {
    throw_at_limit_resource<std::invalid_argument> primary_mr{100};
    throw_at_limit_resource<rmm::out_of_memory> alternate_mr{1000};
    RmmFallbackResource<rmm::out_of_memory> mr{primary_mr, alternate_mr};

    // Check that only `rmm::out_of_memory` exceptions are caught
    EXPECT_THROW(mr.allocate(200), std::invalid_argument);
}
