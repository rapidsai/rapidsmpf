/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <variant>
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

TEST(ReservationEventBuffer, RecordingIsDisabledByDefault) {
    auto br = make_resource();
    br->set_memory_limit(MemoryType::DEVICE, 7);
    EXPECT_TRUE(br->drain_reservation_events().empty());
    EXPECT_TRUE(br->stop_reservation_event_recording().empty());
}

TEST(ReservationEventBuffer, StartRejectsAnActiveSession) {
    auto br = make_resource();
    br->start_reservation_event_recording();
    EXPECT_THROW(br->start_reservation_event_recording(), std::logic_error);
    EXPECT_TRUE(br->stop_reservation_event_recording().empty());

    br->start_reservation_event_recording();
    br->set_memory_limit(MemoryType::DEVICE, 3);
    auto events = br->stop_reservation_event_recording();
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].sequence, 0);
}

TEST(ReservationEventBuffer, RequestAndReservationLifecycleOwnsPayloads) {
    auto br = make_resource();
    br->start_reservation_event_recording();
    auto attributes =
        std::make_shared<ReservationAttributeMap const>(ReservationAttributeMap{
            {"string", std::string{"value"}},
            {"integer", std::int64_t{42}},
            {"boolean", true},
            {"floating", 1.5},
        });

    auto reservation = br->reserve_or_fail(100, MemoryType::DEVICE, attributes);
    auto child = reservation.split(40);
    br->release(child, 10);
    child.clear();
    reservation.clear();
    attributes.reset();

    auto events = br->stop_reservation_event_recording();
    ASSERT_EQ(events.size(), 8);
    for (std::size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].sequence, i);
        EXPECT_GT(events[i].timestamp_ns, 0);
        if (i > 0) {
            EXPECT_GT(events[i].timestamp_ns, events[i - 1].timestamp_ns);
        }
    }

    auto const& submitted = std::get<RequestSubmittedEvent>(events[0].data);
    EXPECT_EQ(submitted.id, 1);
    EXPECT_EQ(submitted.size, 100);
    EXPECT_EQ(
        submitted.memory_types,
        (std::vector<ReservationEventMemoryType>{ReservationEventMemoryType::DEVICE})
    );
    EXPECT_EQ(submitted.policy, ReservationRequestPolicy::IMMEDIATE);
    EXPECT_FALSE(submitted.net_memory_delta.has_value());
    EXPECT_EQ(submitted.attributes.size(), 4);
    EXPECT_EQ(std::get<std::string>(submitted.attributes.at("string")), "value");
    EXPECT_EQ(std::get<std::int64_t>(submitted.attributes.at("integer")), 42);
    EXPECT_TRUE(std::get<bool>(submitted.attributes.at("boolean")));
    EXPECT_DOUBLE_EQ(std::get<double>(submitted.attributes.at("floating")), 1.5);

    auto const& created = std::get<ReservationCreatedEvent>(events[1].data);
    EXPECT_EQ(created.memory_type, ReservationEventMemoryType::DEVICE);
    EXPECT_EQ(created.size, 100);
    EXPECT_EQ(created.request_id, submitted.id);
    EXPECT_EQ(created.parent_id, 0);
    EXPECT_EQ(created.attributes, submitted.attributes);

    auto const& resolved = std::get<RequestResolvedEvent>(events[2].data);
    EXPECT_EQ(resolved.outcome, ReservationRequestOutcome::GRANTED);
    EXPECT_EQ(resolved.reservation_id, created.id);
    EXPECT_EQ(resolved.memory_type, ReservationEventMemoryType::DEVICE);
    EXPECT_EQ(resolved.size, 100);
    EXPECT_EQ(resolved.overbooked_bytes, 0);
    EXPECT_FALSE(resolved.timed_out);
    EXPECT_TRUE(resolved.error.empty());

    auto const& parent_resize = std::get<ReservationResizedEvent>(events[3].data);
    EXPECT_EQ(parent_resize.id, created.id);
    EXPECT_EQ(parent_resize.size, 60);
    EXPECT_EQ(parent_resize.reason, ReservationResizeReason::SPLIT_OFF);

    auto const& child_created = std::get<ReservationCreatedEvent>(events[4].data);
    EXPECT_NE(child_created.id, created.id);
    EXPECT_EQ(child_created.parent_id, created.id);
    EXPECT_EQ(child_created.request_id, 0);
    EXPECT_EQ(child_created.attributes, submitted.attributes);

    auto const& child_resize = std::get<ReservationResizedEvent>(events[5].data);
    EXPECT_EQ(child_resize.id, child_created.id);
    EXPECT_EQ(child_resize.size, 30);
    EXPECT_EQ(child_resize.reason, ReservationResizeReason::CONSUMED);
    EXPECT_EQ(
        std::get<ReservationReleasedEvent>(events[6].data).reason,
        ReservationReleaseReason::CLEARED
    );
    EXPECT_EQ(
        std::get<ReservationReleasedEvent>(events[7].data).reason,
        ReservationReleaseReason::CLEARED
    );
}

TEST(ReservationEventBuffer, EmptyFailedAndLimitEventsPreserveValues) {
    auto br = make_resource(0);
    br->start_reservation_event_recording();

    auto [empty, overbooked] = br->reserve(MemoryType::DEVICE, 10, AllowOverbooking::NO);
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(overbooked, 10);
    br->set_memory_limit(MemoryType::DEVICE, -7);
    EXPECT_THROW(
        static_cast<void>(br->reserve_or_fail(1, MemoryType::PINNED_HOST)),
        std::runtime_error
    );

    auto events = br->stop_reservation_event_recording();
    ASSERT_EQ(events.size(), 5);
    EXPECT_EQ(
        std::get<RequestResolvedEvent>(events[1].data).outcome,
        ReservationRequestOutcome::EMPTY
    );
    auto const& limit = std::get<MemoryLimitChangedEvent>(events[2].data);
    EXPECT_EQ(limit.memory_type, ReservationEventMemoryType::DEVICE);
    EXPECT_EQ(limit.configured_limit, -7);
    auto const& failed = std::get<RequestResolvedEvent>(events[4].data);
    EXPECT_EQ(failed.outcome, ReservationRequestOutcome::FAILED);
    EXPECT_NE(failed.error.find("failed to reserve memory"), std::string::npos);
}

TEST(ReservationEventBuffer, ZeroByteOperationsEmitNothing) {
    auto br = make_resource();
    br->start_reservation_event_recording();
    auto [reservation, overbooked] =
        br->reserve(MemoryType::DEVICE, 0, AllowOverbooking::YES);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_EQ(overbooked, 0);
    EXPECT_TRUE(br->drain_reservation_events().empty());
    EXPECT_TRUE(br->drain_reservation_events().empty());
    EXPECT_TRUE(br->stop_reservation_event_recording().empty());
}

TEST(ReservationEventBuffer, ConcurrentProducersAndDrainsLoseNoEvents) {
    auto br = make_resource();
    br->start_reservation_event_recording();
    constexpr std::size_t producer_count = 4;
    constexpr std::size_t events_per_producer = 250;
    std::atomic<std::size_t> producers_done{0};
    std::vector<std::thread> producers;
    for (std::size_t producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&, producer] {
            for (std::size_t i = 0; i < events_per_producer; ++i) {
                br->set_memory_limit(
                    MemoryType::DEVICE,
                    static_cast<std::int64_t>(producer * events_per_producer + i)
                );
            }
            ++producers_done;
        });
    }

    ReservationEventBatch events;
    while (producers_done.load() != producer_count) {
        auto batch = br->drain_reservation_events();
        events.insert(
            events.end(),
            std::make_move_iterator(batch.begin()),
            std::make_move_iterator(batch.end())
        );
        std::this_thread::yield();
    }
    for (auto& producer : producers) {
        producer.join();
    }
    auto final = br->stop_reservation_event_recording();
    events.insert(
        events.end(),
        std::make_move_iterator(final.begin()),
        std::make_move_iterator(final.end())
    );

    ASSERT_EQ(events.size(), producer_count * events_per_producer);
    for (std::size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].sequence, i);
        if (i > 0) {
            EXPECT_GT(events[i].timestamp_ns, events[i - 1].timestamp_ns);
        }
        EXPECT_TRUE(std::holds_alternative<MemoryLimitChangedEvent>(events[i].data));
    }
    br->set_memory_limit(MemoryType::DEVICE, 99);
    EXPECT_TRUE(br->drain_reservation_events().empty());
}

}  // namespace
