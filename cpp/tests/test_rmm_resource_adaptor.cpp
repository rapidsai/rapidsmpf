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
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/rmm_resource_adaptor.hpp>


using namespace rapidsmpf;

/**
 * @brief User-defined literal for specifying memory sizes in MiB.
 */
constexpr std::size_t operator"" _MiB(unsigned long long val) {
    return val * (1ull << 20);
}

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

TEST(RmmResourceAdaptor, TracksAllocationsAcrossResources) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{1_MiB};
    throw_at_limit_resource<rmm::out_of_memory> fallback_mr{4_MiB};
    RmmResourceAdaptor mr{primary_mr, fallback_mr};

    EXPECT_EQ(mr.current_allocated(), 0);

    void* p1 = mr.allocate(1_MiB);
    EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{p1});
    EXPECT_TRUE(fallback_mr.allocs.empty());
    EXPECT_EQ(mr.current_allocated(), 1_MiB);

    mr.deallocate(p1, 1_MiB);
    EXPECT_TRUE(primary_mr.allocs.empty());
    EXPECT_EQ(mr.current_allocated(), 0);

    void* p2 = mr.allocate(2_MiB);
    EXPECT_TRUE(primary_mr.allocs.empty());
    EXPECT_EQ(fallback_mr.allocs, std::unordered_set<void*>{p2});
    EXPECT_EQ(mr.current_allocated(), 2_MiB);

    mr.deallocate(p2, 2_MiB);
    EXPECT_TRUE(fallback_mr.allocs.empty());
    EXPECT_EQ(mr.current_allocated(), 0);
}

TEST(RmmResourceAdaptor, NoFallbackUsedIfNotNecessary) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{4_MiB};
    throw_at_limit_resource<rmm::out_of_memory> fallback_mr{8_MiB};
    RmmResourceAdaptor mr{primary_mr, fallback_mr};

    void* ptr = mr.allocate(1_MiB);
    EXPECT_EQ(primary_mr.allocs.count(ptr), 1);
    EXPECT_TRUE(fallback_mr.allocs.empty());

    mr.deallocate(ptr, 1_MiB);
}

TEST(RmmResourceAdaptor, NoFallbackProvidedThrowsOnOOM) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{1_MiB};
    RmmResourceAdaptor mr{primary_mr};

    EXPECT_THROW(mr.allocate(8_MiB), rmm::out_of_memory);
}

TEST(RmmResourceAdaptor, RejectsNonOutOfMemoryExceptions) {
    throw_at_limit_resource<std::logic_error> primary_mr{1_MiB};
    throw_at_limit_resource<rmm::out_of_memory> fallback_mr{8_MiB};
    RmmResourceAdaptor mr{primary_mr, fallback_mr};

    EXPECT_THROW(mr.allocate(2_MiB), std::logic_error);
    EXPECT_TRUE(fallback_mr.allocs.empty());
}

TEST(RmmResourceAdaptor, RecordReflectsCorrectStatistics) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{1_MiB};
    throw_at_limit_resource<rmm::out_of_memory> fallback_mr{4_MiB};
    RmmResourceAdaptor mr{primary_mr, fallback_mr};

    auto record_before = mr.get_record();
    EXPECT_EQ(record_before.num_allocs(), 0);
    EXPECT_EQ(record_before.current(), 0);
    EXPECT_EQ(record_before.total(), 0);
    EXPECT_EQ(record_before.peak(), 0);

    // Allocate from primary
    void* p1 = mr.allocate(1_MiB);
    auto record_after_p1 = mr.get_record();

    EXPECT_EQ(record_after_p1.num_allocs(ScopedMemoryRecord::AllocType::Primary), 1);
    EXPECT_EQ(record_after_p1.num_allocs(), 1);
    EXPECT_EQ(record_after_p1.current(ScopedMemoryRecord::AllocType::Primary), 1_MiB);
    EXPECT_EQ(record_after_p1.current(), 1_MiB);
    EXPECT_EQ(record_after_p1.total(ScopedMemoryRecord::AllocType::Primary), 1_MiB);
    EXPECT_EQ(record_after_p1.total(), 1_MiB);
    EXPECT_EQ(record_after_p1.peak(ScopedMemoryRecord::AllocType::Primary), 1_MiB);
    EXPECT_EQ(record_after_p1.peak(), 1_MiB);

    mr.deallocate(p1, 1_MiB);
    auto record_after_d1 = mr.get_record();
    EXPECT_EQ(record_after_d1.current(ScopedMemoryRecord::AllocType::Primary), 0);
    EXPECT_EQ(record_after_d1.current(), 0);
    EXPECT_EQ(record_after_d1.peak(), 1_MiB);  // Peak remains

    // Allocate from fallback
    void* p2 = mr.allocate(2_MiB);
    auto record_after_p2 = mr.get_record();

    EXPECT_EQ(record_after_p2.num_allocs(ScopedMemoryRecord::AllocType::Fallback), 1);
    EXPECT_EQ(record_after_p2.num_allocs(), 2);  // Primary + Fallback
    EXPECT_EQ(record_after_p2.current(ScopedMemoryRecord::AllocType::Fallback), 2_MiB);
    EXPECT_EQ(record_after_p2.current(), 2_MiB);
    EXPECT_EQ(record_after_p2.total(ScopedMemoryRecord::AllocType::Fallback), 2_MiB);
    EXPECT_EQ(record_after_p2.total(), 3_MiB);
    EXPECT_EQ(record_after_p2.peak(ScopedMemoryRecord::AllocType::Fallback), 2_MiB);
    EXPECT_EQ(record_after_p2.peak(), 2_MiB);

    mr.deallocate(p2, 2_MiB);
    auto record_final = mr.get_record();
    EXPECT_EQ(record_final.current(ScopedMemoryRecord::AllocType::Fallback), 0);
    EXPECT_EQ(record_final.current(), 0);
    EXPECT_EQ(record_final.num_allocs(), 2);
    EXPECT_EQ(record_final.total(), 3_MiB);
    EXPECT_EQ(record_final.peak(), 2_MiB);  // Should be the max peak reached
}
