/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <condition_variable>
#include <mutex>
#include <optional>
#include <ostream>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "utils.hpp"


using namespace rapidsmpf;
using HeadroomResult = SpillManager::HeadroomResult;

namespace rapidsmpf {

// Found by gtest through argument-dependent lookup, so a failed comparison prints the
// fields rather than the struct's raw bytes.
inline void PrintTo(SpillManager::HeadroomResult const& result, std::ostream* os) {
    *os << "{deficit=" << result.deficit << ", spilled=" << result.spilled << "}";
}

}  // namespace rapidsmpf

namespace {

/// @brief A buffer resource whose device memory sits exactly at its limit.
///
/// Any device reservation overbooks, and a spill function simulates freeing memory by
/// raising the returned limit. No periodic spill thread, so only the test spills.
std::pair<std::shared_ptr<BufferResource>, std::int64_t> br_at_device_limit() {
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 0}},
        /* periodic_spill_check = */ std::nullopt
    );
    auto const limit =
        safe_cast<std::int64_t>(br->device_mr_adaptor().current_allocated());
    br->set_memory_limit(MemoryType::DEVICE, limit);
    return {std::move(br), limit};
}

}  // namespace

TEST(SpillManager, SpillFunction) {
    // Drive available device memory by adjusting the DEVICE limit at runtime.
    // No real allocations occur in this test, so memory_available equals the
    // currently configured limit.
    std::int64_t mem_available = 10_KiB;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
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
    auto const available = br->spill_manager().spill_to_make_headroom(10_KiB);
    EXPECT_EQ(available, (HeadroomResult{.deficit = 0, .spilled = 0}));
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 40_KiB);

    // If the headroom isn't there, we should spill to get the headroom.
    auto const short_of = br->spill_manager().spill_to_make_headroom(100_KiB);
    EXPECT_EQ(short_of, (HeadroomResult{.deficit = 60_KiB, .spilled = 60_KiB}));
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);

    // A negative headroom is allowed.
    auto const negative = br->spill_manager().spill_to_make_headroom(-100_KiB);
    EXPECT_EQ(negative, (HeadroomResult{.deficit = 0, .spilled = 0}));
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);
}

TEST(SpillManager, HeadroomAccountsForReservations) {
    // As in `SpillFunction`, availability is driven by the DEVICE limit since no real
    // allocations occur.
    std::int64_t mem_available = 100_KiB;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, mem_available}}
    );
    SpillManager::SpillFunction func =
        [&br, &mem_available](std::size_t amount) -> std::size_t {
        mem_available += safe_cast<std::int64_t>(amount);
        br->set_memory_limit(MemoryType::DEVICE, mem_available);
        return amount;
    };
    br->spill_manager().add_spill_function(func, /* priority = */ 0);

    // Without a reservation, a headroom equal to the availability doesn't spill.
    auto const unreserved = br->spill_manager().spill_to_make_headroom(100_KiB);
    EXPECT_EQ(unreserved, (HeadroomResult{.deficit = 0, .spilled = 0}));

    // Reserving 40 KiB leaves the availability untouched but 40 KiB less reservable,
    // and the same headroom now spills that amount.
    auto [reservation, overbooking] =
        br->reserve(MemoryType::DEVICE, 40_KiB, AllowOverbooking::NO);
    EXPECT_EQ(overbooking, 0);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);
    EXPECT_EQ(br->memory_available_for_reservation(MemoryType::DEVICE), 60_KiB);
    auto const reserved = br->spill_manager().spill_to_make_headroom(100_KiB);
    EXPECT_EQ(reserved, (HeadroomResult{.deficit = 40_KiB, .spilled = 40_KiB}));
    EXPECT_EQ(br->memory_available_for_reservation(MemoryType::DEVICE), 100_KiB);
}

TEST(SpillManager, HeadroomReportsTheShortfallNotWhatWasSpilled) {
    auto [br, limit] = br_at_device_limit();

    // Frees 1 KiB however much it is asked for.
    auto const fid = br->spill_manager().add_spill_function(
        [&](std::size_t) -> std::size_t {
            limit += 1_KiB;
            br->set_memory_limit(MemoryType::DEVICE, limit);
            return 1_KiB;
        },
        /* priority = */ 0
    );

    auto const result = br->spill_manager().spill_to_make_headroom(10_KiB);
    EXPECT_EQ(result, (HeadroomResult{.deficit = 10_KiB, .spilled = 1_KiB}));

    br->spill_manager().remove_spill_function(fid);
}

TEST(SpillManager, ReserveAndSpillCreditsAConcurrentSpill) {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered{false};
    bool released{false};

    auto [br, limit] = br_at_device_limit();

    // Frees 8 KiB whenever it runs, whatever it is asked for, which is what spilling in
    // whole buffers looks like. The first call blocks so the test can overlap it.
    std::vector<std::size_t> asks;
    auto const fid = br->spill_manager().add_spill_function(
        [&](std::size_t amount) -> std::size_t {
            {
                std::unique_lock lock(mutex);
                asks.push_back(amount);
                if (!entered) {
                    entered = true;
                    cv.notify_all();
                    cv.wait(lock, [&] { return released; });
                }
            }
            limit += 8_KiB;
            br->set_memory_limit(MemoryType::DEVICE, limit);
            return 8_KiB;
        },
        /* priority = */ 0
    );

    // Hold a spill open, so the reservation below has to queue behind it.
    std::thread spiller{[&] { std::ignore = br->spill_manager().spill(1_KiB); }};
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return entered; });
    }

    // Overbooks by 4 KiB, then waits on the spill lock.
    std::thread reserver{[&] {
        std::ignore = br->reserve_device_memory_and_spill(4_KiB, AllowOverbooking::NO);
    }};
    // Release only once the reservation has made available memory negative, otherwise
    // the reserver finds the memory already free and the test exercises nothing.
    while (br->memory_available_for_reservation(MemoryType::DEVICE) >= 0) {
        std::this_thread::yield();
    }
    {
        std::lock_guard lock(mutex);
        released = true;
    }
    cv.notify_all();
    spiller.join();
    reserver.join();
    br->spill_manager().remove_spill_function(fid);

    // The blocked spill freed 8 KiB against 4 KiB of overbooking, so the reservation
    // needed nothing further and the spill function ran once, not twice.
    EXPECT_EQ(asks.size(), 1u);
    EXPECT_EQ(asks.front(), 1_KiB);
}

TEST(SpillManager, ReserveAndSpillIgnoresAConcurrentCallersOverbooking) {
    auto [br, limit] = br_at_device_limit();

    // Frees exactly what it is asked for. Then another caller's overbooked reservation
    // lands, after this call's spill and before it judges whether the spill was enough.
    std::optional<MemoryReservation> other;
    auto const fid = br->spill_manager().add_spill_function(
        [&](std::size_t amount) -> std::size_t {
            limit += safe_cast<std::int64_t>(amount);
            br->set_memory_limit(MemoryType::DEVICE, limit);
            if (!other.has_value()) {
                other.emplace(
                    br->reserve(MemoryType::DEVICE, 8_KiB, AllowOverbooking::YES).first
                );
            }
            return amount;
        },
        /* priority = */ 0
    );

    // This call's own overbooking was covered in full. The other caller's is not its to
    // answer for, so it must not fail on account of it.
    EXPECT_NO_THROW(
        std::ignore = br->reserve_device_memory_and_spill(4_KiB, AllowOverbooking::NO)
    );

    br->spill_manager().remove_spill_function(fid);
}

TEST(SpillManager, ReserveAndSpillIgnoresAnEarlierCallersOverbooking) {
    auto [br, limit] = br_at_device_limit();

    // Only 4 KiB is spillable, enough for one caller's overbooking and not two.
    std::size_t spillable{4_KiB};
    auto const fid = br->spill_manager().add_spill_function(
        [&](std::size_t amount) -> std::size_t {
            auto const freed = std::min(amount, spillable);
            spillable -= freed;
            limit += safe_cast<std::int64_t>(freed);
            br->set_memory_limit(MemoryType::DEVICE, limit);
            return freed;
        },
        /* priority = */ 0
    );

    // Another caller has already overbooked by 4 KiB, so `reserve()` reports this call's
    // overbooking as 8 KiB. Half of that is not this call's to cover.
    auto earlier = br->reserve(MemoryType::DEVICE, 4_KiB, AllowOverbooking::YES).first;
    EXPECT_NO_THROW(
        std::ignore = br->reserve_device_memory_and_spill(4_KiB, AllowOverbooking::NO)
    );

    br->spill_manager().remove_spill_function(fid);
}
