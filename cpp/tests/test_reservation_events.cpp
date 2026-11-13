/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>

using namespace rapidsmpf;

namespace {

std::shared_ptr<BufferResource> make_resource(std::int64_t device_limit = 1024) {
    return BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, device_limit}}
    );
}

TEST(ReservationEvents, RecordingLifecycleAndExactDeltas) {
    auto br = make_resource();
    auto const context = ReservationContextCreator{}.create();
    EXPECT_TRUE(br->drain_reservation_events().events.empty());
    EXPECT_TRUE(br->stop_reservation_event_recording().events.empty());

    br->start_reservation_event_recording();
    EXPECT_THROW(br->start_reservation_event_recording(), std::logic_error);
    auto reservation = br->reserve_or_fail(100, MemoryType::DEVICE, context);
    br->release(reservation, 25);
    reservation.clear();

    auto batch = br->stop_reservation_event_recording();
    ASSERT_EQ(batch.events.size(), 3);
    EXPECT_EQ(batch.dropped_event_count, 0);
    EXPECT_EQ(batch.events[0].reservation_context, context);
    EXPECT_EQ(batch.events[0].memory_type, MemoryType::DEVICE);
    EXPECT_EQ(batch.events[0].byte_delta, 100);
    EXPECT_EQ(batch.events[1].byte_delta, -25);
    EXPECT_EQ(batch.events[2].byte_delta, -75);
    for (std::size_t i = 0; i < batch.events.size(); ++i) {
        EXPECT_GT(batch.events[i].timestamp_ns, 0);
        if (i > 0) {
            EXPECT_GE(batch.events[i].timestamp_ns, batch.events[i - 1].timestamp_ns);
        }
    }
}

TEST(ReservationEvents, SplitAndMoveConserveOccupancy) {
    auto br = make_resource();
    auto const context = ReservationContextCreator{}.create();
    br->start_reservation_event_recording();
    auto reservation = br->reserve_or_fail(100, MemoryType::DEVICE, context);
    auto child = reservation.split(40);
    auto moved = std::move(child);
    EXPECT_EQ(moved.reservation_context(), context);
    EXPECT_TRUE(br->drain_reservation_events().events.size() == 1);

    moved.clear();
    reservation.clear();
    auto batch = br->stop_reservation_event_recording();
    ASSERT_EQ(batch.events.size(), 2);
    EXPECT_EQ(batch.events[0].byte_delta, -40);
    EXPECT_EQ(batch.events[1].byte_delta, -60);
}

TEST(ReservationEvents, OnlySameSessionEmitsNegativeDelta) {
    auto br = make_resource();
    auto before =
        br->reserve_or_fail(10, MemoryType::DEVICE, ReservationContextCreator{}.create());

    br->start_reservation_event_recording();
    before.clear();
    auto observed =
        br->reserve_or_fail(20, MemoryType::DEVICE, ReservationContextCreator{}.create());
    auto first = br->stop_reservation_event_recording();
    ASSERT_EQ(first.events.size(), 1);
    EXPECT_EQ(first.events[0].byte_delta, 20);

    br->start_reservation_event_recording();
    observed.clear();
    auto second = br->stop_reservation_event_recording();
    EXPECT_TRUE(second.events.empty());
}

TEST(ReservationEvents, ZeroByteOperationsEmitNothing) {
    auto br = make_resource();
    br->start_reservation_event_recording();
    auto [reservation, overbooked] = br->reserve(
        MemoryType::DEVICE, 0, AllowOverbooking::YES, ReservationContextCreator{}.create()
    );
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_EQ(overbooked, 0);
    auto batch = br->stop_reservation_event_recording();
    EXPECT_TRUE(batch.events.empty());
    EXPECT_EQ(batch.dropped_event_count, 0);
}

std::shared_ptr<BufferResource> make_capped_resource(std::size_t max_reservation_events) {
    return BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 1024}},
        std::chrono::milliseconds{1},
        std::make_shared<StreamPool>(1),
        Statistics::disabled(),
        std::nullopt,
        std::chrono::milliseconds{100},
        AllowOverbooking::YES,
        max_reservation_events
    );
}

TEST(ReservationEvents, MaxEventsDropsAndCountsExcess) {
    auto br = make_capped_resource(2);
    EXPECT_EQ(br->max_reservation_events(), 2);
    br->start_reservation_event_recording();
    auto reservation = br->reserve_or_fail(100, MemoryType::DEVICE);
    br->release(reservation, 25);
    br->release(reservation, 25);  // Dropped.
    reservation.clear();  // Dropped.

    auto drained = br->drain_reservation_events();
    ASSERT_EQ(drained.events.size(), 2);
    EXPECT_EQ(drained.dropped_event_count, 2);
    EXPECT_EQ(drained.events[0].byte_delta, 100);
    EXPECT_EQ(drained.events[1].byte_delta, -25);

    // Draining frees space, and the dropped count was reset.
    auto reservation2 = br->reserve_or_fail(10, MemoryType::DEVICE);
    auto final = br->stop_reservation_event_recording();
    ASSERT_EQ(final.events.size(), 1);
    EXPECT_EQ(final.dropped_event_count, 0);
}

TEST(ReservationEvents, DroppedReservationAlsoSkipsItsRelease) {
    auto br = make_capped_resource(1);
    br->start_reservation_event_recording();
    auto recorded = br->reserve_or_fail(10, MemoryType::DEVICE);
    auto dropped = br->reserve_or_fail(20, MemoryType::DEVICE);  // Cap reached.

    auto first = br->drain_reservation_events();
    ASSERT_EQ(first.events.size(), 1);
    EXPECT_EQ(first.events[0].byte_delta, 10);
    EXPECT_EQ(first.dropped_event_count, 1);

    // The drain freed space, but `dropped` was never observed by the session, so
    // releasing it must not record a negative delta without a matching reservation.
    dropped.clear();
    recorded.clear();
    auto final = br->stop_reservation_event_recording();
    ASSERT_EQ(final.events.size(), 1);
    EXPECT_EQ(final.events[0].byte_delta, -10);
    EXPECT_EQ(final.dropped_event_count, 0);
}

TEST(ReservationEvents, MaxEventsFromOptions) {
    EXPECT_EQ(max_reservation_events_from_options(config::Options{}), 1'000'000);
    using Map = std::unordered_map<std::string, std::string>;
    config::Options options{Map{{"max_reservation_events", "7"}}};
    EXPECT_EQ(max_reservation_events_from_options(options), 7);
    EXPECT_THROW(
        max_reservation_events_from_options(
            config::Options{Map{{"max_reservation_events", "-1"}}}
        ),
        std::invalid_argument
    );
}

TEST(ReservationEvents, ConcurrentProducersAndDrainsLoseNoEvents) {
    auto br = make_resource(1 << 20);
    br->start_reservation_event_recording();
    constexpr std::size_t producer_count = 4;
    constexpr std::size_t events_per_producer = 200;
    std::atomic<std::size_t> producers_done{0};
    std::vector<std::thread> producers;
    for (std::size_t producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&, producer] {
            for (std::size_t i = 0; i < events_per_producer; ++i) {
                auto reservation = br->reserve_or_fail(
                    1, MemoryType::DEVICE, ReservationContextCreator{}.create()
                );
                reservation.clear();
            }
            ++producers_done;
        });
    }

    std::vector<ReservationEvent> events;
    while (producers_done.load() != producer_count) {
        auto batch = br->drain_reservation_events();
        events.insert(
            events.end(),
            std::make_move_iterator(batch.events.begin()),
            std::make_move_iterator(batch.events.end())
        );
        std::this_thread::yield();
    }
    for (auto& producer : producers) {
        producer.join();
    }
    auto final = br->stop_reservation_event_recording();
    events.insert(
        events.end(),
        std::make_move_iterator(final.events.begin()),
        std::make_move_iterator(final.events.end())
    );

    ASSERT_EQ(events.size(), 2 * producer_count * events_per_producer);
    for (std::size_t i = 1; i < events.size(); ++i) {
        EXPECT_GE(events[i].timestamp_ns, events[i - 1].timestamp_ns);
    }
}

TEST(ReservationEvents, ContextsAreOpaqueCopyableAndUnique) {
    static_assert(std::is_copy_constructible_v<ReservationContext>);
    static_assert(std::is_copy_assignable_v<ReservationContext>);
    static_assert(!std::is_constructible_v<ReservationContext, std::uint64_t>);

    auto creator = ReservationContextCreator{};
    auto first = creator.create();
    auto copy = first;
    auto second = creator.create();

    EXPECT_EQ(first, copy);
    EXPECT_NE(first, second);
    EXPECT_TRUE(first.is_attributed());
    EXPECT_FALSE(ReservationContext{}.is_attributed());
}

}  // namespace
