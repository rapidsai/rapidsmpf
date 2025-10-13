/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <barrier>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

template <typename ExceptionType>
struct throw_at_limit_resource final : public rmm::mr::device_memory_resource {
    throw_at_limit_resource(std::size_t limit) : limit{limit} {}

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        if (bytes > limit) {
            throw ExceptionType{"foo"};
        }
        void* ptr{nullptr};
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMallocAsync(&ptr, bytes, stream));
        allocs.insert(ptr);
        return ptr;
    }

    void do_deallocate(
        void* ptr, std::size_t, rmm::cuda_stream_view stream
    ) noexcept override {
        RAPIDSMPF_ASSERT_CUDA_SUCCESS(cudaFreeAsync(ptr, stream.value()));
        allocs.erase(ptr);
    }

    [[nodiscard]] bool do_is_equal(
        rmm::mr::device_memory_resource const& other
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

    auto main_record_before = mr.get_main_record();
    EXPECT_EQ(main_record_before.num_total_allocs(), 0);
    EXPECT_EQ(main_record_before.current(), 0);
    EXPECT_EQ(main_record_before.total(), 0);
    EXPECT_EQ(main_record_before.peak(), 0);

    // Allocate from primary
    void* p1 = mr.allocate(1_MiB);
    auto main_record_after_p1 = mr.get_main_record();

    EXPECT_EQ(
        main_record_after_p1.num_total_allocs(ScopedMemoryRecord::AllocType::PRIMARY), 1
    );
    EXPECT_EQ(main_record_after_p1.num_total_allocs(), 1);
    EXPECT_EQ(
        main_record_after_p1.current(ScopedMemoryRecord::AllocType::PRIMARY), 1_MiB
    );
    EXPECT_EQ(main_record_after_p1.current(), 1_MiB);
    EXPECT_EQ(main_record_after_p1.total(ScopedMemoryRecord::AllocType::PRIMARY), 1_MiB);
    EXPECT_EQ(main_record_after_p1.total(), 1_MiB);
    EXPECT_EQ(main_record_after_p1.peak(ScopedMemoryRecord::AllocType::PRIMARY), 1_MiB);
    EXPECT_EQ(main_record_after_p1.peak(), 1_MiB);

    mr.deallocate(p1, 1_MiB);
    auto main_record_after_d1 = mr.get_main_record();
    EXPECT_EQ(main_record_after_d1.current(ScopedMemoryRecord::AllocType::PRIMARY), 0);
    EXPECT_EQ(main_record_after_d1.current(), 0);
    EXPECT_EQ(main_record_after_d1.peak(), 1_MiB);  // Peak remains

    // Allocate from fallback
    void* p2 = mr.allocate(2_MiB);
    auto main_record_after_p2 = mr.get_main_record();

    EXPECT_EQ(
        main_record_after_p2.num_total_allocs(ScopedMemoryRecord::AllocType::FALLBACK), 1
    );
    EXPECT_EQ(main_record_after_p2.num_total_allocs(), 2);  // PRIMARY + FALLBACK
    EXPECT_EQ(
        main_record_after_p2.current(ScopedMemoryRecord::AllocType::FALLBACK), 2_MiB
    );
    EXPECT_EQ(main_record_after_p2.current(), 2_MiB);
    EXPECT_EQ(main_record_after_p2.total(ScopedMemoryRecord::AllocType::FALLBACK), 2_MiB);
    EXPECT_EQ(main_record_after_p2.total(), 3_MiB);
    EXPECT_EQ(main_record_after_p2.peak(ScopedMemoryRecord::AllocType::FALLBACK), 2_MiB);
    EXPECT_EQ(main_record_after_p2.peak(), 2_MiB);

    mr.deallocate(p2, 2_MiB);
    auto main_record_final = mr.get_main_record();
    EXPECT_EQ(main_record_final.current(ScopedMemoryRecord::AllocType::FALLBACK), 0);
    EXPECT_EQ(main_record_final.current(), 0);
    EXPECT_EQ(main_record_final.num_total_allocs(), 2);
    EXPECT_EQ(main_record_final.total(), 3_MiB);
    EXPECT_EQ(main_record_final.peak(), 2_MiB);  // Should be the max peak reached
}

TEST(ScopedMemoryRecord, AddSubscopeMergesNestedScopeCorrectly) {
    ScopedMemoryRecord parent;
    ScopedMemoryRecord subscope;

    // Parent: Allocate and deallocate
    parent.record_allocation(
        ScopedMemoryRecord::AllocType::PRIMARY, 300
    );  // parant-peak: 300
    parent.record_allocation(
        ScopedMemoryRecord::AllocType::FALLBACK, 400
    );  // parant-peak: 400
    parent.record_deallocation(
        ScopedMemoryRecord::AllocType::PRIMARY, 20
    );  // parant-current: 280
    parent.record_deallocation(
        ScopedMemoryRecord::AllocType::FALLBACK, 50
    );  // parant-current: 350

    // Subscope: Allocate and deallocate
    subscope.record_allocation(
        ScopedMemoryRecord::AllocType::PRIMARY, 100
    );  // parant-peak: 280+100, child-peak: 100
    subscope.record_allocation(
        ScopedMemoryRecord::AllocType::PRIMARY, 50
    );  // parant-peak: 280+150, child-peak: 150
    subscope.record_allocation(
        ScopedMemoryRecord::AllocType::FALLBACK, 200
    );  // parant-peak: 350+200, child-peak: 200
    subscope.record_deallocation(
        ScopedMemoryRecord::AllocType::PRIMARY, 30
    );  // child-current: 120
    subscope.record_deallocation(
        ScopedMemoryRecord::AllocType::FALLBACK, 80
    );  // child-current: 120

    // Merge subscope into parent
    parent.add_subscope(subscope);

    // Expect current (after merge) is sum of currents
    EXPECT_EQ(parent.current(ScopedMemoryRecord::AllocType::PRIMARY), 280 + 120);
    EXPECT_EQ(parent.current(ScopedMemoryRecord::AllocType::FALLBACK), 350 + 120);

    // Expect totals to accumulate
    EXPECT_EQ(parent.total(ScopedMemoryRecord::AllocType::PRIMARY), 300 + 150);
    EXPECT_EQ(parent.total(ScopedMemoryRecord::AllocType::FALLBACK), 400 + 200);

    // Alloc count
    EXPECT_EQ(parent.num_total_allocs(ScopedMemoryRecord::AllocType::PRIMARY), 3);
    EXPECT_EQ(parent.num_total_allocs(ScopedMemoryRecord::AllocType::FALLBACK), 2);

    // Corrected peak logic: parent current at time of merge + subscope peak
    EXPECT_EQ(parent.peak(ScopedMemoryRecord::AllocType::PRIMARY), 280 + 150);
    EXPECT_EQ(parent.peak(ScopedMemoryRecord::AllocType::FALLBACK), 350 + 200);
}

TEST(ScopedMemoryRecord, AddScopeMergesSiblingScopesCorrectly) {
    ScopedMemoryRecord scope1;
    ScopedMemoryRecord scope2;

    // Simulate allocations in scope1
    scope1.record_allocation(ScopedMemoryRecord::AllocType::PRIMARY, 100);
    scope1.record_allocation(ScopedMemoryRecord::AllocType::FALLBACK, 200);
    // Deallocate from scope1
    scope1.record_deallocation(ScopedMemoryRecord::AllocType::PRIMARY, 30);
    scope1.record_deallocation(ScopedMemoryRecord::AllocType::FALLBACK, 50);

    // Simulate allocations in scope2
    scope2.record_allocation(ScopedMemoryRecord::AllocType::PRIMARY, 50);
    scope2.record_allocation(ScopedMemoryRecord::AllocType::FALLBACK, 400);
    // Deallocate from scope2 (note: large dealloc triggers negative current)
    scope2.record_deallocation(ScopedMemoryRecord::AllocType::FALLBACK, 600);

    // Merge scope2 into scope1 as peer scope
    scope1.add_scope(scope2);

    // Peaks should be max of peaks from each scope
    EXPECT_EQ(scope1.peak(ScopedMemoryRecord::AllocType::PRIMARY), std::max(100, 50));
    EXPECT_EQ(scope1.peak(ScopedMemoryRecord::AllocType::FALLBACK), std::max(200, 400));

    // Currents are summed
    // PRIMARY: 100 - 30 + 50 = 120
    EXPECT_EQ(scope1.current(ScopedMemoryRecord::AllocType::PRIMARY), 120);
    // FALLBACK: 200 - 50 + 400 - 600 = -50
    EXPECT_EQ(scope1.current(ScopedMemoryRecord::AllocType::FALLBACK), -50);

    // Totals are additive
    EXPECT_EQ(scope1.total(ScopedMemoryRecord::AllocType::PRIMARY), 100 + 50);
    EXPECT_EQ(scope1.total(ScopedMemoryRecord::AllocType::FALLBACK), 200 + 400);

    // Allocation counts are summed
    EXPECT_EQ(scope1.num_total_allocs(ScopedMemoryRecord::AllocType::PRIMARY), 2);
    EXPECT_EQ(scope1.num_total_allocs(ScopedMemoryRecord::AllocType::FALLBACK), 2);
}

TEST(RmmResourceAdaptor, EmptyScopedMemoryRecord) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    mr.begin_scoped_memory_record();
    auto scope = mr.end_scoped_memory_record();
    EXPECT_EQ(scope.current(), 0);
    EXPECT_EQ(scope.total(), 0);
    EXPECT_EQ(scope.peak(), 0);
    EXPECT_EQ(scope.num_total_allocs(), 0);
}

TEST(RmmResourceAdaptorScopedMemory, SingleScopedAllocationTracksCorrectly) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    mr.begin_scoped_memory_record();
    void* p = mr.allocate(1_MiB);
    auto scope = mr.end_scoped_memory_record();

    EXPECT_EQ(scope.current(), 1_MiB);
    EXPECT_EQ(scope.total(), 1_MiB);
    EXPECT_EQ(scope.peak(), 1_MiB);
    EXPECT_EQ(scope.num_total_allocs(), 1);

    mr.deallocate(p, 1_MiB);
}

TEST(RmmResourceAdaptorScopedMemory, NestedScopedAllocationsMerged) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    mr.begin_scoped_memory_record();  // Outer

    void* p1 = mr.allocate(1_MiB);  // Alloc in outer

    mr.begin_scoped_memory_record();  // Inner
    void* p2 = mr.allocate(2_MiB);  // Alloc in inner
    auto inner = mr.end_scoped_memory_record();

    auto outer = mr.end_scoped_memory_record();  // Merge inner into outer

    // Inner record
    EXPECT_EQ(inner.num_total_allocs(), 1);
    EXPECT_EQ(inner.total(), 2_MiB);

    // Outer should reflect both outer + inner allocations
    EXPECT_EQ(outer.num_total_allocs(), 2);
    EXPECT_EQ(outer.total(), 3_MiB);
    EXPECT_EQ(outer.peak(), 3_MiB);

    mr.deallocate(p2, 2_MiB);
    mr.deallocate(p1, 1_MiB);
}

TEST(RmmResourceAdaptorScopedMemory, NestedScopedTracksAllocsAndDeallocs) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    mr.begin_scoped_memory_record();  // Outer

    void* p1 = mr.allocate(1_MiB);
    void* p2 = mr.allocate(2_MiB);
    mr.deallocate(p2, 2_MiB);

    mr.begin_scoped_memory_record();  // Inner
    void* p3 = mr.allocate(3_MiB);
    mr.deallocate(p3, 3_MiB);
    auto inner = mr.end_scoped_memory_record();

    auto outer = mr.end_scoped_memory_record();

    // Outer: p1 allocated, p2 allocated + deallocated, p3 (via inner)
    EXPECT_EQ(inner.num_total_allocs(), 1);
    EXPECT_EQ(inner.total(), 3_MiB);

    EXPECT_EQ(outer.num_total_allocs(), 3);
    EXPECT_EQ(outer.total(), 1_MiB + 2_MiB + 3_MiB);
    EXPECT_EQ(outer.current(), 1_MiB);  // Only p1 is left

    mr.deallocate(p1, 1_MiB);
}

TEST(RmmResourceAdaptorScopedMemory, NestedDeallocationYieldsNegativeStats) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    // Allocate in outer scope
    mr.begin_scoped_memory_record();  // Outer
    void* p = mr.allocate(1_MiB);

    // Begin nested scope
    mr.begin_scoped_memory_record();  // Inner
    mr.deallocate(p, 1_MiB);  // Dealloc done in inner scope
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

TEST(RmmResourceAdaptorScopedMemory, MultiThreadedScopedAllocations) {
    constexpr int num_threads = 8;
    constexpr int num_allocs_per_thread = 8;
    constexpr std::size_t alloc_size = 1_MiB;

    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
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
                void* ptr = mr.allocate(alloc_size);
                allocations[i].push_back(ptr);
            }

            // Deallocate some (but not all) to test `current()` accounting
            for (int j = 0; j < num_allocs_per_thread / 2; ++j) {
                mr.deallocate(allocations[i][j], alloc_size);
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
            mr.deallocate(allocations[i][j], alloc_size);
        }
    }
    EXPECT_EQ(mr.current_allocated(), 0);  // All allocations have been released
}

TEST(RmmResourceAdaptorScopedMemory, CrossThreadNestedScopesNotMerged) {
    constexpr std::size_t outer_alloc_size = 1_MiB;
    constexpr std::size_t inner_alloc_size = 2_MiB;

    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    void* outer_alloc = nullptr;
    void* inner_alloc = nullptr;
    rapidsmpf::ScopedMemoryRecord inner_record;

    mr.begin_scoped_memory_record();  // Outer scope in main thread

    outer_alloc = mr.allocate(outer_alloc_size);

    std::thread t([&]() {
        mr.begin_scoped_memory_record();  // Inner scope in different thread
        inner_alloc = mr.allocate(inner_alloc_size);
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
    mr.deallocate(inner_alloc, inner_alloc_size);
    mr.deallocate(outer_alloc, outer_alloc_size);
}
