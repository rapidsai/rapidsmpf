/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "utils.hpp"


using namespace rapidsmpf;

TEST(SpillManager, SpillFunction) {
    // Drive available device memory by adjusting the DEVICE limit at runtime.
    // No real allocations occur in this test, so memory_available equals the
    // currently configured limit.
    std::int64_t mem_available = 10_KiB;
    auto br = BufferResource::create(
        cudf::get_current_device_resource_ref(),
        rapidsmpf::PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE, mem_available}}
    );
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);

    // Spill function that increases the available memory perfectly.
    SpillManager::SpillFunction func1 =
        [&br, &mem_available](std::size_t amount) -> std::size_t {
        mem_available += safe_cast<std::int64_t>(amount);
        br->set_memory_limit(MemoryType::DEVICE, mem_available);
        return amount;
    };
    br->spill_manager().add_spill_function(func1, /* priority = */ 1);
    EXPECT_EQ(br->spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 20_KiB);

    // Spill function that never spill any memory but has a higher priority.
    bool func2_called = false;
    SpillManager::SpillFunction func2 = [&func2_called](std::size_t) -> std::size_t {
        func2_called = true;
        return 0;
    };
    auto fid2 = br->spill_manager().add_spill_function(func2, /* priority = */ 2);
    EXPECT_EQ(br->spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_TRUE(func2_called);
    func2_called = false;
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 30_KiB);

    // Removing `func2` means it shouldn't run.
    br->spill_manager().remove_spill_function(fid2);
    EXPECT_EQ(br->spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_FALSE(func2_called);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 40_KiB);

    // If the headroom is already there, no spilling should be happening.
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(10_KiB), 0);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 40_KiB);

    // If the headroom isn't there, we should spill to get the headroom.
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(100_KiB), 60_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);

    // A negative headroom is allowed.
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(-100_KiB), 0);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);
}

// Verify that multiple concurrent `spill()` calls execute the spill function in
// parallel. The spill function blocks at a rendezvous point until all worker
// threads have entered it; with an exclusive lock this would deadlock and the
// test would time out.
TEST(SpillManager, ConcurrentSpill) {
    constexpr int num_threads = 4;
    constexpr auto rendezvous_timeout = std::chrono::seconds(5);

    BufferResource br{
        cudf::get_current_device_resource_ref(),
        rapidsmpf::PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE, []() -> std::int64_t { return 0; }}}
    };

    std::mutex m;
    std::condition_variable cv;
    int entered = 0;

    SpillManager::SpillFunction func = [&](std::size_t amount) -> std::size_t {
        std::unique_lock<std::mutex> lock(m);
        ++entered;
        if (entered == num_threads) {
            cv.notify_all();
        }
        // If spill() calls run in parallel, all threads quickly reach
        // `entered == num_threads` and proceed. If they're serialized, every
        // thread but the last times out here, so a successful (non-timeout)
        // wait_for is proof that the calls actually overlapped.
        EXPECT_TRUE(cv.wait_for(lock, rendezvous_timeout, [&] {
            return entered == num_threads;
        })) << "spill() calls did not run concurrently";
        return amount;
    };

    br.spill_manager().add_spill_function(func, /* priority = */ 0);

    std::vector<std::future<std::size_t>> futures;
    futures.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&] {
            return br.spill_manager().spill(1_KiB);
        }));
    }
    auto const total_spilled = std::accumulate(
        futures.begin(),
        futures.end(),
        std::size_t{0},
        [](std::size_t sum, std::future<std::size_t>& f) { return sum + f.get(); }
    );

    EXPECT_EQ(total_spilled, num_threads * 1_KiB);
}
