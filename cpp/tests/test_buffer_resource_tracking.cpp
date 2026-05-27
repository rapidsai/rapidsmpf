/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <barrier>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

template <typename ExceptionType>
struct throw_at_limit_resource_impl {
    explicit throw_at_limit_resource_impl(std::size_t limit) : limit{limit} {}

    throw_at_limit_resource_impl(throw_at_limit_resource_impl const&) = delete;
    throw_at_limit_resource_impl& operator=(throw_at_limit_resource_impl const&) = delete;
    throw_at_limit_resource_impl(throw_at_limit_resource_impl&&) = delete;
    throw_at_limit_resource_impl& operator=(throw_at_limit_resource_impl&&) = delete;

    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t /*alignment*/ = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        if (bytes > limit) {
            throw ExceptionType{"foo"};
        }
        void* ptr{nullptr};
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMallocAsync(&ptr, bytes, stream.get()));
        allocs.insert(ptr);
        return ptr;
    }

    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t /*bytes*/,
        std::size_t /*alignment*/ = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        RAPIDSMPF_CUDA_TRY_FATAL(cudaFreeAsync(ptr, stream.get()));
        allocs.erase(ptr);
    }

    [[nodiscard]] bool operator==(
        throw_at_limit_resource_impl const& other
    ) const noexcept {
        return this == &other;
    }

    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        auto stream = cudaStream_t{nullptr};
        auto ptr = allocate(cuda::stream_ref{stream}, bytes, alignment);
        RAPIDSMPF_CUDA_TRY_FATAL(cudaStreamSynchronize(stream));
        return ptr;
    }

    void deallocate_sync(
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        auto stream = cudaStream_t{nullptr};
        deallocate(cuda::stream_ref{stream}, ptr, bytes, alignment);
        RAPIDSMPF_CUDA_TRY_FATAL(cudaStreamSynchronize(stream));
    }

    friend void get_property(
        throw_at_limit_resource_impl const&, cuda::mr::device_accessible
    ) noexcept {}

    std::size_t limit;
    std::unordered_set<void*> allocs{};
};

template <typename ExceptionType>
struct throw_at_limit_resource
    : public cuda::mr::shared_resource<throw_at_limit_resource_impl<ExceptionType>> {
    using impl_type = throw_at_limit_resource_impl<ExceptionType>;
    using shared_base = cuda::mr::shared_resource<impl_type>;

    explicit throw_at_limit_resource(std::size_t limit)
        : shared_base(cuda::mr::make_shared_resource<impl_type>(limit)) {}

    friend void get_property(
        throw_at_limit_resource const&, cuda::mr::device_accessible
    ) noexcept {}

    [[nodiscard]] std::unordered_set<void*> const& allocs() const {
        return this->get().allocs;
    }
};

TEST(BufferResourceTracking, TracksAllocations) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{4_MiB};
    BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{primary_mr}};

    EXPECT_EQ(mr.current_allocated(), 0);

    void* p1 = mr.allocate_sync(1_MiB);
    EXPECT_EQ(primary_mr.allocs(), std::unordered_set<void*>{p1});
    EXPECT_EQ(mr.current_allocated(), 1_MiB);

    mr.deallocate_sync(p1, 1_MiB);
    EXPECT_TRUE(primary_mr.allocs().empty());
    EXPECT_EQ(mr.current_allocated(), 0);
}

TEST(BufferResourceTracking, OOMPropagates) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{1_MiB};
    BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{primary_mr}};

    EXPECT_THROW((void)mr.allocate_sync(8_MiB), rmm::out_of_memory);
}

TEST(BufferResourceTracking, PropagatesNonOutOfMemoryExceptions) {
    throw_at_limit_resource<std::logic_error> primary_mr{1_MiB};
    BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{primary_mr}};

    EXPECT_THROW(std::ignore = mr.allocate_sync(2_MiB), std::logic_error);
}

TEST(BufferResourceTracking, RecordReflectsCorrectStatistics) {
    throw_at_limit_resource<rmm::out_of_memory> primary_mr{4_MiB};
    BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{primary_mr}};

    auto main_record_before = mr.get_main_record();
    EXPECT_EQ(main_record_before.num_total_allocs(), 0);
    EXPECT_EQ(main_record_before.current(), 0);
    EXPECT_EQ(main_record_before.total(), 0);
    EXPECT_EQ(main_record_before.peak(), 0);

    void* p1 = mr.allocate_sync(1_MiB);
    auto main_record_after_p1 = mr.get_main_record();

    EXPECT_EQ(main_record_after_p1.num_total_allocs(), 1);
    EXPECT_EQ(main_record_after_p1.current(), 1_MiB);
    EXPECT_EQ(main_record_after_p1.total(), 1_MiB);
    EXPECT_EQ(main_record_after_p1.peak(), 1_MiB);

    mr.deallocate_sync(p1, 1_MiB);
    auto main_record_after_d1 = mr.get_main_record();
    EXPECT_EQ(main_record_after_d1.current(), 0);
    EXPECT_EQ(main_record_after_d1.peak(), 1_MiB);  // Peak remains

    void* p2 = mr.allocate_sync(2_MiB);
    auto main_record_after_p2 = mr.get_main_record();

    EXPECT_EQ(main_record_after_p2.num_total_allocs(), 2);
    EXPECT_EQ(main_record_after_p2.current(), 2_MiB);
    EXPECT_EQ(main_record_after_p2.total(), 3_MiB);
    EXPECT_EQ(main_record_after_p2.peak(), 2_MiB);

    mr.deallocate_sync(p2, 2_MiB);
    auto main_record_final = mr.get_main_record();
    EXPECT_EQ(main_record_final.current(), 0);
    EXPECT_EQ(main_record_final.num_total_allocs(), 2);
    EXPECT_EQ(main_record_final.total(), 3_MiB);
    EXPECT_EQ(main_record_final.peak(), 2_MiB);  // Should be the max peak reached
}

TEST(ScopedMemoryRecord, AddSubscopeMergesNestedScopeCorrectly) {
    ScopedMemoryRecord parent;
    ScopedMemoryRecord subscope;

    // Parent: allocate then partially deallocate.
    parent.record_allocation(300);
    parent.record_deallocation(20);

    // Subscope: allocate then partially deallocate.
    subscope.record_allocation(100);
    subscope.record_allocation(50);
    subscope.record_deallocation(30);

    // Merge subscope into parent
    parent.add_subscope(subscope);

    EXPECT_EQ(parent.current(), 280 + 120);
    EXPECT_EQ(parent.total(), 300 + 150);
    EXPECT_EQ(parent.num_total_allocs(), 3);
    // Peak: parent current at merge time + subscope peak.
    EXPECT_EQ(parent.peak(), 280 + 150);
}

TEST(ScopedMemoryRecord, AddScopeMergesSiblingScopesCorrectly) {
    ScopedMemoryRecord scope1;
    ScopedMemoryRecord scope2;

    scope1.record_allocation(100);
    scope1.record_deallocation(30);

    scope2.record_allocation(50);

    // Merge scope2 into scope1 as a peer scope.
    scope1.add_scope(scope2);

    EXPECT_EQ(scope1.peak(), std::max(100, 50));
    // 100 - 30 + 50 = 120
    EXPECT_EQ(scope1.current(), 120);
    EXPECT_EQ(scope1.total(), 100 + 50);
    EXPECT_EQ(scope1.num_total_allocs(), 2);
}

TEST(BufferResourceTracking, EmptyScopedMemoryRecord) {
    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};

    mr.begin_scoped_memory_record();
    auto scope = mr.end_scoped_memory_record();
    EXPECT_EQ(scope.current(), 0);
    EXPECT_EQ(scope.total(), 0);
    EXPECT_EQ(scope.peak(), 0);
    EXPECT_EQ(scope.num_total_allocs(), 0);
}

TEST(BufferResourceTrackingScopedMemory, SingleScopedAllocationTracksCorrectly) {
    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};

    mr.begin_scoped_memory_record();
    void* p = mr.allocate_sync(1_MiB);
    auto scope = mr.end_scoped_memory_record();

    EXPECT_EQ(scope.current(), 1_MiB);
    EXPECT_EQ(scope.total(), 1_MiB);
    EXPECT_EQ(scope.peak(), 1_MiB);
    EXPECT_EQ(scope.num_total_allocs(), 1);

    mr.deallocate_sync(p, 1_MiB);
}

TEST(BufferResourceTrackingScopedMemory, NestedScopedAllocationsMerged) {
    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};

    mr.begin_scoped_memory_record();  // Outer

    void* p1 = mr.allocate_sync(1_MiB);  // Alloc in outer

    mr.begin_scoped_memory_record();  // Inner
    void* p2 = mr.allocate_sync(2_MiB);  // Alloc in inner
    auto inner = mr.end_scoped_memory_record();

    auto outer = mr.end_scoped_memory_record();  // Merge inner into outer

    // Inner record
    EXPECT_EQ(inner.num_total_allocs(), 1);
    EXPECT_EQ(inner.total(), 2_MiB);

    // Outer should reflect both outer + inner allocations
    EXPECT_EQ(outer.num_total_allocs(), 2);
    EXPECT_EQ(outer.total(), 3_MiB);
    EXPECT_EQ(outer.peak(), 3_MiB);

    mr.deallocate_sync(p2, 2_MiB);
    mr.deallocate_sync(p1, 1_MiB);
}

TEST(BufferResourceTrackingScopedMemory, NestedScopedTracksAllocsAndDeallocs) {
    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};

    mr.begin_scoped_memory_record();  // Outer

    void* p1 = mr.allocate_sync(1_MiB);
    void* p2 = mr.allocate_sync(2_MiB);
    mr.deallocate_sync(p2, 2_MiB);

    mr.begin_scoped_memory_record();  // Inner
    void* p3 = mr.allocate_sync(3_MiB);
    mr.deallocate_sync(p3, 3_MiB);
    auto inner = mr.end_scoped_memory_record();

    auto outer = mr.end_scoped_memory_record();

    // Outer: p1 allocated, p2 allocated + deallocated, p3 (via inner)
    EXPECT_EQ(inner.num_total_allocs(), 1);
    EXPECT_EQ(inner.total(), 3_MiB);

    EXPECT_EQ(outer.num_total_allocs(), 3);
    EXPECT_EQ(outer.total(), 1_MiB + 2_MiB + 3_MiB);
    EXPECT_EQ(outer.current(), 1_MiB);  // Only p1 is left

    mr.deallocate_sync(p1, 1_MiB);
}

TEST(BufferResourceTrackingScopedMemory, NestedDeallocationYieldsNegativeStats) {
    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};

    // Allocate in outer scope
    mr.begin_scoped_memory_record();  // Outer
    void* p = mr.allocate_sync(1_MiB);

    // Begin nested scope
    mr.begin_scoped_memory_record();  // Inner
    mr.deallocate_sync(p, 1_MiB);  // Dealloc done in inner scope
    auto inner = mr.end_scoped_memory_record();

    auto outer = mr.end_scoped_memory_record();

    // Inner scope should show a negative current value and no allocations
    EXPECT_EQ(inner.num_total_allocs(), 0);
    EXPECT_EQ(inner.current(), -static_cast<std::int64_t>(1_MiB));
    EXPECT_EQ(inner.total(), 0);

    // Outer scope had one alloc, and a dealloc performed by inner
    EXPECT_EQ(outer.num_total_allocs(), 1);
    EXPECT_EQ(outer.current(), 0);  // Net usage is zero
}

TEST(BufferResourceTrackingScopedMemory, MultiThreadedScopedAllocations) {
    constexpr int num_threads = 8;
    constexpr int num_allocs_per_thread = 8;
    constexpr std::size_t alloc_size = 1_MiB;

    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> allocations(num_threads);
    std::vector<rapidsmpf::ScopedMemoryRecord> records(num_threads);
    std::barrier barrier(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // Wait until all threads are ready to start
            barrier.arrive_and_wait();

            mr.begin_scoped_memory_record();

            // Perform multiple allocations
            for (int j = 0; j < num_allocs_per_thread; ++j) {
                void* ptr = mr.allocate_sync(alloc_size);
                allocations[i].push_back(ptr);
            }

            // Deallocate some (but not all) to test `current()` accounting
            for (int j = 0; j < num_allocs_per_thread / 2; ++j) {
                mr.deallocate_sync(allocations[i][j], alloc_size);
            }

            records[i] = mr.end_scoped_memory_record();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Check that each thread's memory record matches expectations
    for (int i = 0; i < num_threads; ++i) {
        const auto& rec = records[i];

        EXPECT_EQ(rec.num_total_allocs(), num_allocs_per_thread);
        EXPECT_EQ(rec.total(), alloc_size * num_allocs_per_thread);
        EXPECT_EQ(rec.peak(), alloc_size * num_allocs_per_thread);
        EXPECT_EQ(
            rec.current(), alloc_size * (num_allocs_per_thread / 2)
        );  // Half still allocated

        // Now deallocate the remaining allocations
        for (int j = num_allocs_per_thread / 2; j < num_allocs_per_thread; ++j) {
            mr.deallocate_sync(allocations[i][j], alloc_size);
        }
    }
    EXPECT_EQ(mr.current_allocated(), 0);  // All allocations have been released
}

TEST(BufferResourceTracking, EqualityWithCudaMemoryResource) {
    rmm::mr::cuda_memory_resource cuda_mr{};

    BufferResource adaptor_a{
        cuda::mr::any_resource<cuda::mr::device_accessible>{cuda_mr}
    };
    BufferResource adaptor_b{
        cuda::mr::any_resource<cuda::mr::device_accessible>{cuda_mr}
    };

    // Both wrap same resouce but have difference shared states
    EXPECT_NE(adaptor_a, adaptor_b);

    // A copy shares the same control block -> equal.
    BufferResource adaptor_a_copy = adaptor_a;
    EXPECT_EQ(adaptor_a, adaptor_a_copy);
}

TEST(BufferResourceTrackingScopedMemory, CrossThreadNestedScopesNotMerged) {
    constexpr std::size_t outer_alloc_size = 1_MiB;
    constexpr std::size_t inner_alloc_size = 2_MiB;

    rapidsmpf::BufferResource mr{cuda::mr::any_resource<cuda::mr::device_accessible>{
        cudf::get_current_device_resource_ref()
    }};
    void* outer_alloc = nullptr;
    void* inner_alloc = nullptr;
    rapidsmpf::ScopedMemoryRecord inner_record;

    mr.begin_scoped_memory_record();  // Outer scope in main thread

    outer_alloc = mr.allocate_sync(outer_alloc_size);

    std::thread t([&]() {
        mr.begin_scoped_memory_record();  // Inner scope in different thread
        inner_alloc = mr.allocate_sync(inner_alloc_size);
        inner_record = mr.end_scoped_memory_record();
    });

    t.join();

    auto outer_record = mr.end_scoped_memory_record();  // End outer scope

    // Outer scope should not include inner thread's allocation
    EXPECT_EQ(outer_record.num_total_allocs(), 1);
    EXPECT_EQ(outer_record.total(), outer_alloc_size);
    EXPECT_EQ(outer_record.current(), outer_alloc_size);
    EXPECT_EQ(outer_record.peak(), outer_alloc_size);

    // Inner record should reflect its own allocation
    EXPECT_EQ(inner_record.num_total_allocs(), 1);
    EXPECT_EQ(inner_record.total(), inner_alloc_size);
    EXPECT_EQ(inner_record.current(), inner_alloc_size);
    EXPECT_EQ(inner_record.peak(), inner_alloc_size);

    // Clean up
    mr.deallocate_sync(inner_alloc, inner_alloc_size);
    mr.deallocate_sync(outer_alloc, outer_alloc_size);
}
