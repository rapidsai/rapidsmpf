/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>

using namespace rapidsmpf;

namespace {

struct RecorderState {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::string> events;
    std::vector<std::uint64_t> sequences;
    int retains{0};
    int releases{0};
    bool block_limit{false};
    bool in_limit{false};
    std::vector<std::int64_t> limits;
    std::vector<std::size_t> submitted_attribute_counts;
    std::vector<std::size_t> created_attribute_counts;
    std::set<std::uint32_t> submitted_attribute_types;
};

void retain(void* user) noexcept {
    static_cast<RecorderState*>(user)->retains++;
}

void release(void* user) noexcept {
    static_cast<RecorderState*>(user)->releases++;
}

void add(RecorderState& state, std::uint64_t sequence, std::string event) {
    std::lock_guard lock(state.mutex);
    state.sequences.push_back(sequence);
    state.events.push_back(std::move(event));
}

void submitted(
    void* user,
    std::uint64_t sequence,
    std::uint64_t,
    std::size_t,
    std::uint32_t const*,
    std::size_t,
    std::uint32_t,
    bool,
    std::int64_t,
    rapidsmpf_reservation_attributes_view attributes
) noexcept {
    auto& state = *static_cast<RecorderState*>(user);
    {
        std::lock_guard lock(state.mutex);
        state.submitted_attribute_counts.push_back(attributes.size);
        for (std::size_t i = 0; i < attributes.size; ++i) {
            state.submitted_attribute_types.insert(attributes.data[i].type);
        }
    }
    add(state, sequence, "submitted");
}

void resolved(
    void* user,
    std::uint64_t sequence,
    std::uint64_t,
    std::uint32_t outcome,
    std::uint64_t,
    std::uint32_t,
    std::size_t,
    std::size_t,
    bool,
    rapidsmpf_string_view
) noexcept {
    static constexpr char const* names[] = {"granted", "empty", "failed", "cancelled"};
    add(*static_cast<RecorderState*>(user),
        sequence,
        outcome < std::size(names) ? names[outcome] : "unknown");
}

void created(
    void* user,
    std::uint64_t sequence,
    std::uint64_t,
    std::uint32_t,
    std::size_t,
    rapidsmpf_reservation_attributes_view attributes,
    std::uint64_t,
    std::uint64_t parent
) noexcept {
    auto& state = *static_cast<RecorderState*>(user);
    {
        std::lock_guard lock(state.mutex);
        state.created_attribute_counts.push_back(attributes.size);
    }
    add(state, sequence, parent == 0 ? "created" : "split-created");
}

void resized(
    void* user, std::uint64_t sequence, std::uint64_t, std::size_t, std::uint32_t reason
) noexcept {
    add(*static_cast<RecorderState*>(user),
        sequence,
        reason == RAPIDSMPF_RESERVATION_SPLIT_OFF ? "split-off" : "consumed");
}

void released_reservation(
    void* user, std::uint64_t sequence, std::uint64_t, std::uint32_t reason
) noexcept {
    add(*static_cast<RecorderState*>(user),
        sequence,
        reason == RAPIDSMPF_RESERVATION_CLEARED ? "cleared" : "released");
}

void limit_changed(
    void* user, std::uint64_t sequence, std::uint32_t, std::int64_t limit
) noexcept {
    auto& state = *static_cast<RecorderState*>(user);
    std::unique_lock lock(state.mutex);
    state.sequences.push_back(sequence);
    state.events.push_back("limit");
    state.limits.push_back(limit);
    state.in_limit = true;
    state.cv.notify_all();
    state.cv.wait(lock, [&] { return !state.block_limit; });
}

rapidsmpf_reservation_recorder recorder(RecorderState& state) {
    rapidsmpf_reservation_recorder ret{};
    ret.abi_version = RAPIDSMPF_RESERVATION_RECORDER_ABI_VERSION;
    ret.struct_size = sizeof(ret);
    ret.user = &state;
    ret.retain_user = retain;
    ret.release_user = release;
    ret.request_submitted = submitted;
    ret.request_resolved = resolved;
    ret.reservation_created = created;
    ret.reservation_resized = resized;
    ret.reservation_released = released_reservation;
    ret.memory_limit_changed = limit_changed;
    return ret;
}

TEST(ReservationRecorder, RequestAndReservationLifecycleIsSequenced) {
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 1024}}
    );
    RecorderState state;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);

    auto [reservation, overbooked] =
        br->reserve(MemoryType::DEVICE, 100, AllowOverbooking::NO);
    EXPECT_EQ(overbooked, 0);
    auto child = reservation.split(40);
    br->release(child, 10);
    child.clear();
    reservation.clear();

    EXPECT_EQ(
        state.events,
        (std::vector<std::string>{
            "submitted",
            "created",
            "granted",
            "split-off",
            "split-created",
            "consumed",
            "cleared",
            "cleared"
        })
    );
    for (std::size_t i = 0; i < state.sequences.size(); ++i) {
        EXPECT_EQ(state.sequences[i], i);
    }
    EXPECT_NE(reservation.id(), child.id());
    br->set_reservation_recorder(nullptr);
    EXPECT_EQ(state.retains, 1);
    EXPECT_EQ(state.releases, 1);
}

TEST(ReservationRecorder, ZeroByteOperationsEmitNothing) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    RecorderState state;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);
    auto [reservation, overbooked] =
        br->reserve(MemoryType::DEVICE, 0, AllowOverbooking::YES);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_EQ(overbooked, 0);
    EXPECT_TRUE(state.events.empty());
}

TEST(ReservationRecorder, PublicPoliciesHaveOneTerminalOutcome) {
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 0}, {MemoryType::HOST, 1024}}
    );
    RecorderState state;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);

    auto [empty, _] = br->reserve(MemoryType::DEVICE, 10, AllowOverbooking::NO);
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(state.events, (std::vector<std::string>{"submitted", "empty"}));

    state.events.clear();
    std::array const types{MemoryType::DEVICE, MemoryType::HOST};
    auto reservation = br->reserve_or_fail(10, types);
    EXPECT_EQ(reservation.mem_type(), MemoryType::HOST);
    EXPECT_EQ(
        state.events, (std::vector<std::string>{"submitted", "created", "granted"})
    );
}

TEST(ReservationRecorder, AttributesReachRequestsAndReservations) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    RecorderState state;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);
    auto attributes =
        std::make_shared<ReservationAttributeMap const>(ReservationAttributeMap{
            {"string", std::string{"value"}},
            {"integer", std::int64_t{42}},
            {"boolean", true},
            {"floating", 1.5},
        });

    auto reservation = br->reserve_or_fail(10, MemoryType::HOST, attributes);

    EXPECT_EQ(state.submitted_attribute_counts, (std::vector<std::size_t>{4}));
    EXPECT_EQ(state.created_attribute_counts, (std::vector<std::size_t>{4}));
    EXPECT_EQ(
        state.submitted_attribute_types,
        (std::set<std::uint32_t>{
            RAPIDSMPF_RESERVATION_ATTRIBUTE_STRING,
            RAPIDSMPF_RESERVATION_ATTRIBUTE_I64,
            RAPIDSMPF_RESERVATION_ATTRIBUTE_BOOL,
            RAPIDSMPF_RESERVATION_ATTRIBUTE_F64,
        })
    );
    EXPECT_EQ(reservation.attributes(), attributes);
}

TEST(ReservationRecorder, ReplacementWaitsForInflightCallback) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    RecorderState state;
    state.block_limit = true;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);

    std::thread emitter([&] { br->set_memory_limit(MemoryType::DEVICE, 7); });
    {
        std::unique_lock lock(state.mutex);
        state.cv.wait(lock, [&] { return state.in_limit; });
    }
    std::atomic<bool> replacement_returned{false};
    std::thread replacement([&] {
        br->set_reservation_recorder(nullptr);
        replacement_returned = true;
    });
    std::this_thread::yield();
    EXPECT_FALSE(replacement_returned);
    {
        std::lock_guard lock(state.mutex);
        state.block_limit = false;
    }
    state.cv.notify_all();
    emitter.join();
    replacement.join();
    EXPECT_TRUE(replacement_returned);
    EXPECT_EQ(state.releases, 1);
}

TEST(ReservationRecorder, MemoryLimitPreservesNegativeValue) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    RecorderState state;
    auto callbacks = recorder(state);
    br->set_reservation_recorder(&callbacks);

    br->set_memory_limit(MemoryType::DEVICE, -7);

    EXPECT_EQ(br->memory_limit(MemoryType::DEVICE), -7);
    EXPECT_EQ(state.events, (std::vector<std::string>{"limit"}));
    EXPECT_EQ(state.limits, (std::vector<std::int64_t>{-7}));
}

}  // namespace
