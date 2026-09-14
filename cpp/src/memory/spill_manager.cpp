/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <utility>
#include <variant>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

namespace {

std::int64_t steady_clock_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()
    )
        .count();
}

std::uint64_t event_id(SpillEvent const& event) {
    return std::visit([](auto const& record) { return record.event_id; }, event);
}

void set_event_id(SpillEvent& event, std::uint64_t id) {
    std::visit([id](auto& record) { record.event_id = id; }, event);
}

}  // namespace

class SpillManager::SpillAttemptScope {
  public:
    SpillAttemptScope(SpillManager& manager, ActiveSpillAttempt attempt)
        : manager_{manager} {
        std::lock_guard lock(manager_.active_attempt_mutex_);
        previous_ = std::move(manager_.active_attempt_);
        manager_.active_attempt_ = std::move(attempt);
    }

    ~SpillAttemptScope() {
        std::lock_guard lock(manager_.active_attempt_mutex_);
        manager_.active_attempt_ = std::move(previous_);
    }

  private:
    SpillManager& manager_;
    std::optional<ActiveSpillAttempt> previous_;
};

void SpillEventCollector::enable(std::size_t capacity) {
    RAPIDSMPF_EXPECTS(capacity > 0, "spill event capacity must be greater than zero");
    std::lock_guard lock(mutex_);
    events_.clear();
    events_.resize(capacity);
    oldest_index_ = 0;
    size_ = 0;
    dropped_events_.store(0, std::memory_order_release);
    enabled_.store(true, std::memory_order_release);
}

void SpillEventCollector::disable() noexcept {
    enabled_.store(false, std::memory_order_release);
    std::lock_guard lock(mutex_);
    events_.clear();
    oldest_index_ = 0;
    size_ = 0;
}

bool SpillEventCollector::enabled() const noexcept {
    return enabled_.load(std::memory_order_acquire);
}

std::uint64_t SpillEventCollector::sequence() const noexcept {
    return next_sequence_.load(std::memory_order_acquire);
}

std::uint64_t SpillEventCollector::dropped_events() const noexcept {
    return dropped_events_.load(std::memory_order_acquire);
}

std::vector<SpillEvent> SpillEventCollector::read(
    std::uint64_t begin_sequence, std::uint64_t end_sequence
) const {
    std::vector<SpillEvent> ret;
    std::lock_guard lock(mutex_);
    ret.reserve(std::min(size_, events_.size()));
    for (std::size_t i = 0; i < size_; ++i) {
        auto const& event = events_[(oldest_index_ + i) % events_.size()];
        auto const id = event_id(event);
        if (begin_sequence <= id && id < end_sequence) {
            ret.push_back(event);
        }
    }
    return ret;
}

std::vector<SpillAttemptRecord> SpillEventCollector::read_attempts(
    std::uint64_t begin_sequence, std::uint64_t end_sequence
) const {
    std::vector<SpillAttemptRecord> ret;
    for (auto const& event : read(begin_sequence, end_sequence)) {
        if (auto const* attempt = std::get_if<SpillAttemptRecord>(&event)) {
            ret.push_back(*attempt);
        }
    }
    return ret;
}

std::vector<SpillTransferRecord> SpillEventCollector::read_transfers(
    std::uint64_t begin_sequence, std::uint64_t end_sequence
) const {
    std::vector<SpillTransferRecord> ret;
    for (auto const& event : read(begin_sequence, end_sequence)) {
        if (auto const* transfer = std::get_if<SpillTransferRecord>(&event)) {
            ret.push_back(*transfer);
        }
    }
    return ret;
}

void SpillEventCollector::record(SpillEvent event) noexcept {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }
    try {
        std::lock_guard lock(mutex_);
        if (!enabled_.load(std::memory_order_relaxed) || events_.empty()) {
            return;
        }
        set_event_id(event, next_sequence_.fetch_add(1, std::memory_order_acq_rel));
        if (size_ < events_.size()) {
            events_[(oldest_index_ + size_) % events_.size()] = std::move(event);
            ++size_;
        } else {
            events_[oldest_index_] = std::move(event);
            oldest_index_ = (oldest_index_ + 1) % events_.size();
            dropped_events_.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (...) {
        // Telemetry must never replace or suppress an allocation/copy exception.
        dropped_events_.fetch_add(1, std::memory_order_relaxed);
    }
}

SpillBufferOwnerScope::SpillBufferOwnerScope(
    SpillManager& manager, std::optional<SpillAttributionToken> buffer_owner
)
    : manager_{manager} {
    std::lock_guard lock(manager_.active_attempt_mutex_);
    if (manager_.active_attempt_.has_value()
        && manager_.active_attempt_->thread_id == std::this_thread::get_id())
    {
        previous_ = manager_.active_attempt_->buffer_owner;
        manager_.active_attempt_->buffer_owner = std::move(buffer_owner);
        has_active_attempt_ = true;
    }
}

SpillBufferOwnerScope::~SpillBufferOwnerScope() {
    if (has_active_attempt_) {
        std::lock_guard lock(manager_.active_attempt_mutex_);
        manager_.active_attempt_->buffer_owner = previous_;
    }
}

SpillManager::SpillManager(
    BufferResource* br, std::optional<Duration> periodic_spill_check
)
    : br_{br} {
    if (periodic_spill_check.has_value()) {
        periodic_spill_thread_.emplace(
            [this]() {
                spill_to_make_headroom(
                    0, SpillReason::PERIODIC, std::nullopt, /* is_background = */ true
                );
            },
            *periodic_spill_check
        );
    }
}

SpillManager::~SpillManager() {
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->stop();
    }
}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function, int priority
) {
    return add_spill_function(std::move(spill_function), priority, std::nullopt);
}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function,
    int priority,
    std::optional<SpillAttributionToken> buffer_owner
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const id = spill_function_id_counter_++;
    RAPIDSMPF_EXPECTS(
        spill_functions_
            .insert(
                {id,
                 RegisteredSpillFunction{
                     std::move(spill_function), std::move(buffer_owner)
                 }}
            )
            .second,
        "corrupted id counter"
    );
    spill_function_priorities_.insert({priority, id});

    // Make sure the spill thread is running.
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->resume();
    }
    return id;
}

void SpillManager::remove_spill_function(SpillFunctionID fid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& prio = spill_function_priorities_;
    for (auto it = prio.begin(); it != prio.end(); ++it) {
        if (it->second == fid) {
            prio.erase(it);  // Erase the first occurrence
            break;  // Exit after erasing to ensure only the first one is removed
        }
    }
    spill_functions_.erase(fid);

    // Asynchronously pause the spill thread if no spill functions are left.
    if (periodic_spill_thread_.has_value() && spill_functions_.empty()) {
        periodic_spill_thread_->pause_nb();
    }
}

std::size_t SpillManager::spill_unsafe(
    std::size_t amount,
    SpillReason reason,
    std::optional<SpillAttributionToken> evictor,
    bool is_background
) {
    auto const attempt_id = attempt_id_counter_.fetch_add(1, std::memory_order_relaxed);
    event_collector_.record(
        SpillAttemptRecord{
            .attempt_id = attempt_id,
            .start_ns = steady_clock_ns(),
            .requested_bytes = amount,
            .reason = reason,
            .is_background = is_background,
            .evictor = evictor,
        }
    );
    SpillAttemptScope attempt_scope{
        *this,
        ActiveSpillAttempt{
            attempt_id,
            reason,
            is_background,
            std::move(evictor),
            std::nullopt,
            std::this_thread::get_id()
        }
    };

    std::size_t spilled{0};
    for (auto const [_, fid] : spill_function_priorities_) {
        if (spilled >= amount) {
            break;
        }
        auto const& registered = spill_functions_.at(fid);
        SpillBufferOwnerScope owner_scope{*this, registered.buffer_owner};
        spilled += registered.function(amount - spilled);
    }
    return spilled;
}

std::size_t SpillManager::spill_to_make_headroom_unsafe(
    std::int64_t headroom,
    SpillReason reason,
    std::optional<SpillAttributionToken> evictor,
    bool is_background
) {
    // TODO: check other memory types.
    std::int64_t const available =
        br_->memory_available_for_reservation(MemoryType::DEVICE);
    if (headroom <= available) {
        return 0;
    }
    return spill_unsafe(
        safe_cast<std::size_t>(headroom - available),
        reason,
        std::move(evictor),
        is_background
    );
}

std::size_t SpillManager::spill(std::size_t amount) {
    return spill(amount, SpillReason::EXPLICIT, std::nullopt);
}

std::size_t SpillManager::spill(
    std::size_t amount, SpillReason reason, std::optional<SpillAttributionToken> evictor
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_unsafe(amount, reason, std::move(evictor), /* is_background = */ false);
}

std::size_t SpillManager::spill_to_make_headroom(std::int64_t headroom) {
    return spill_to_make_headroom(
        headroom, SpillReason::EXPLICIT, std::nullopt, /* is_background = */ false
    );
}

std::size_t SpillManager::spill_to_make_headroom(
    std::int64_t headroom,
    SpillReason reason,
    std::optional<SpillAttributionToken> evictor,
    bool is_background
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_to_make_headroom_unsafe(
        headroom, reason, std::move(evictor), is_background
    );
}

std::optional<std::size_t> SpillManager::try_spill_to_make_headroom(
    std::int64_t headroom
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return std::nullopt;
    }
    return spill_to_make_headroom_unsafe(
        headroom, SpillReason::EXPLICIT, std::nullopt, /* is_background = */ false
    );
}

SpillEventCollector& SpillManager::event_collector() noexcept {
    return event_collector_;
}

SpillEventCollector const& SpillManager::event_collector() const noexcept {
    return event_collector_;
}

void SpillManager::record_transfer_submission(
    MemoryType source,
    MemoryType destination,
    std::size_t bytes,
    std::int64_t submission_start_ns,
    std::int64_t submission_end_ns,
    bool is_success
) noexcept {
    if (source != MemoryType::DEVICE
        || (destination != MemoryType::PINNED_HOST && destination != MemoryType::HOST))
    {
        return;
    }
    std::optional<ActiveSpillAttempt> attempt;
    {
        std::lock_guard lock(active_attempt_mutex_);
        if (!active_attempt_.has_value()
            || active_attempt_->thread_id != std::this_thread::get_id())
        {
            return;
        }
        attempt = active_attempt_;
    }
    event_collector_.record(
        SpillTransferRecord{
            .attempt_id = attempt->attempt_id,
            .submission_start_ns = submission_start_ns,
            .submission_end_ns = submission_end_ns,
            .submitted_bytes = is_success ? bytes : 0,
            .source = source,
            .destination = destination,
            .reason = attempt->reason,
            .is_background = attempt->is_background,
            .evictor = attempt->evictor,
            .buffer_owner = attempt->buffer_owner,
            .is_success = is_success,
        }
    );
}

}  // namespace rapidsmpf
