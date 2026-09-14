/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

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
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(10_KiB), 0);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 40_KiB);

    // If the headroom isn't there, we should spill to get the headroom.
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(100_KiB), 60_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);

    // A negative headroom is allowed.
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(-100_KiB), 0);
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
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(100_KiB), 0);

    // Reserving 40 KiB leaves the availability untouched but 40 KiB less reservable,
    // and the same headroom now spills that amount.
    auto [reservation, overbooking] =
        br->reserve(MemoryType::DEVICE, 40_KiB, AllowOverbooking::NO);
    EXPECT_EQ(overbooking, 0);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 100_KiB);
    EXPECT_EQ(br->memory_available_for_reservation(MemoryType::DEVICE), 60_KiB);
    EXPECT_EQ(br->spill_manager().spill_to_make_headroom(100_KiB), 40_KiB);
    EXPECT_EQ(br->memory_available_for_reservation(MemoryType::DEVICE), 100_KiB);
}

TEST(SpillManager, TrySpillToMakeHeadroomWhenIdle) {
    std::int64_t mem_available = 0;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, mem_available}},
        std::nullopt  // No periodic spill thread
    );

    // Spill function that increases the available memory perfectly.
    SpillManager::SpillFunction func =
        [&br, &mem_available](std::size_t amount) -> std::size_t {
        mem_available += safe_cast<std::int64_t>(amount);
        br->set_memory_limit(MemoryType::DEVICE, mem_available);
        return amount;
    };
    br->spill_manager().add_spill_function(func, /* priority = */ 1);

    // Nothing else holds the spill lock, so this spills like the blocking version.
    auto const spilled = br->spill_manager().try_spill_to_make_headroom(10_KiB);
    ASSERT_TRUE(spilled.has_value());
    EXPECT_EQ(*spilled, 10_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);
}

TEST(SpillManager, TrySpillToMakeHeadroomSkipsWhileSpilling) {
    std::int64_t mem_available = 0;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, mem_available}},
        std::nullopt  // No periodic spill thread
    );

    std::mutex mutex;
    std::condition_variable cv;
    bool entered_spill{false};
    bool release_spill{false};

    // Spill function that blocks until the test releases it, keeping the spill
    // manager's lock held in the meantime.
    SpillManager::SpillFunction func = [&](std::size_t amount) -> std::size_t {
        {
            std::unique_lock lock(mutex);
            entered_spill = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_spill; });
        }
        mem_available += safe_cast<std::int64_t>(amount);
        br->set_memory_limit(MemoryType::DEVICE, mem_available);
        return amount;
    };
    br->spill_manager().add_spill_function(func, /* priority = */ 1);

    std::thread thd([&] { br->spill_manager().spill_to_make_headroom(10_KiB); });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return entered_spill; });
    }

    // A spill is in progress, so this returns immediately instead of blocking.
    EXPECT_EQ(br->spill_manager().try_spill_to_make_headroom(10_KiB), std::nullopt);

    {
        std::lock_guard lock(mutex);
        release_spill = true;
    }
    cv.notify_all();
    thd.join();
}

TEST(SpillManager, RecordsSpillScopedDeviceToHostSubmission) {
    constexpr std::size_t size = 4_KiB;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 2 * size}},
        std::nullopt
    );
    auto device_reservation = br->reserve_or_fail(size, MemoryType::DEVICE);
    auto buffer =
        br->make_buffer(size, br->stream_pool()->get_stream(), device_reservation);

    auto& collector = br->spill_manager().event_collector();
    collector.enable(16);
    auto const begin = collector.sequence();
    br->spill_manager().add_spill_function(
        [&](std::size_t amount) {
            auto host_reservation = br->reserve_or_fail(amount, MemoryType::HOST);
            buffer = br->move(std::move(buffer), host_reservation);
            return amount;
        },
        /* priority = */ 0,
        SpillAttributionToken{22}
    );

    EXPECT_EQ(
        br->spill_manager().spill(size, SpillReason::EXPLICIT, SpillAttributionToken{11}),
        size
    );
    auto const events = collector.read(begin, collector.sequence());
    ASSERT_EQ(events.size(), 2);

    auto const& attempt = std::get<SpillAttemptRecord>(events[0]);
    auto const& transfer = std::get<SpillTransferRecord>(events[1]);
    EXPECT_EQ(attempt.attempt_id, transfer.attempt_id);
    EXPECT_EQ(attempt.requested_bytes, size);
    EXPECT_EQ(attempt.reason, SpillReason::EXPLICIT);
    EXPECT_EQ(attempt.evictor, SpillAttributionToken{11});
    EXPECT_EQ(transfer.submitted_bytes, size);
    EXPECT_EQ(transfer.source, MemoryType::DEVICE);
    EXPECT_EQ(transfer.destination, MemoryType::HOST);
    EXPECT_EQ(transfer.reason, SpillReason::EXPLICIT);
    EXPECT_EQ(transfer.evictor, SpillAttributionToken{11});
    EXPECT_EQ(transfer.buffer_owner, SpillAttributionToken{22});
    EXPECT_TRUE(transfer.is_success);
    EXPECT_LE(transfer.submission_start_ns, transfer.submission_end_ns);
}

TEST(SpillManager, OrdinaryCopyDoesNotEmitSpillTelemetry) {
    constexpr std::size_t size = 4_KiB;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), PinnedMemoryDisabled, {}, std::nullopt
    );
    auto source = br->make_buffer(
        br->stream_pool()->get_stream(), br->reserve_or_fail(size, MemoryType::DEVICE)
    );
    auto destination = br->make_buffer(
        br->stream_pool()->get_stream(), br->reserve_or_fail(size, MemoryType::HOST)
    );
    auto& collector = br->spill_manager().event_collector();
    collector.enable(16);
    auto const begin = collector.sequence();

    buffer_copy(br->statistics(), *destination, *source, size);

    EXPECT_TRUE(collector.read(begin, collector.sequence()).empty());
}

TEST(SpillManager, ConcurrentOrdinaryMoveIsNotAttributedToActiveSpill) {
    constexpr std::size_t size = 4_KiB;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), PinnedMemoryDisabled, {}, std::nullopt
    );
    auto buffer = br->make_buffer(
        br->stream_pool()->get_stream(), br->reserve_or_fail(size, MemoryType::DEVICE)
    );
    auto& collector = br->spill_manager().event_collector();
    collector.enable(16);
    auto const begin = collector.sequence();

    std::mutex mutex;
    std::condition_variable cv;
    bool callback_entered{false};
    bool release_callback{false};
    br->spill_manager().add_spill_function(
        [&](std::size_t amount) {
            std::unique_lock lock(mutex);
            callback_entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_callback; });
            return amount;
        },
        0
    );
    std::thread spill_thread([&] { br->spill_manager().spill(size); });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return callback_entered; });
    }

    auto host_reservation = br->reserve_or_fail(size, MemoryType::HOST);
    buffer = br->move(std::move(buffer), host_reservation);

    {
        std::lock_guard lock(mutex);
        release_callback = true;
    }
    cv.notify_all();
    spill_thread.join();

    auto const events = collector.read(begin, collector.sequence());
    ASSERT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<SpillAttemptRecord>(events[0]));
}

TEST(SpillManager, CollectorReportsOverflowAndPreservesSequenceOrder) {
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), PinnedMemoryDisabled, {}, std::nullopt
    );
    auto& collector = br->spill_manager().event_collector();
    collector.enable(2);
    auto const begin = collector.sequence();

    EXPECT_EQ(br->spill_manager().spill(0), 0);
    EXPECT_EQ(br->spill_manager().spill(0), 0);
    EXPECT_EQ(br->spill_manager().spill(0), 0);

    auto const events = collector.read(begin, collector.sequence());
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(collector.dropped_events(), 1);
    EXPECT_LT(
        std::get<SpillAttemptRecord>(events[0]).event_id,
        std::get<SpillAttemptRecord>(events[1]).event_id
    );
}
