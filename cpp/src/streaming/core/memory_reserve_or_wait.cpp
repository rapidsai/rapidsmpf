/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <coro/sync_wait.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::streaming {

MemoryReserveOrWait::MemoryReserveOrWait(
    config::Options options,
    std::shared_ptr<Logger> logger,
    MemoryType mem_type,
    std::shared_ptr<CoroThreadPoolExecutor> executor,
    std::shared_ptr<BufferResource> br
)
    : mem_type_{mem_type},
      logger_{std::move(logger)},
      executor_{std::move(executor)},
      br_{std::move(br)},
      timeout_{options.get<Duration>("memory_reserve_timeout", parse_duration)} {
    RAPIDSMPF_EXPECTS(logger_ != nullptr, "logger cannot be NULL");
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");
}

MemoryReserveOrWait::~MemoryReserveOrWait() noexcept {
    coro::sync_wait(shutdown());
}

Actor MemoryReserveOrWait::shutdown() {
    // Move the pending requests and joinable periodic task out under the mutex,
    // then release the lock. Both the queue shutdown and the task await can block
    // or suspend, so they must not run while holding the mutex.
    std::unique_lock lock(mutex_);
    auto reservation_requests = std::move(reservation_requests_);
    auto periodic_memory_check_task =
        std::exchange(periodic_memory_check_task_, std::nullopt);
    lock.unlock();

    // Shut down all request queues so any waiters are unblocked, then wait for
    // the periodic task to exit (if one was running).
    if (!reservation_requests.empty()) {
        std::vector<Actor> actors;
        for (Request const& request : reservation_requests) {
            actors.push_back(request.queue.shutdown());
        }
        coro_results(co_await coro::when_all(std::move(actors)));
    }
    if (periodic_memory_check_task.has_value()) {
        co_await *periodic_memory_check_task;
    }
}

coro::task<MemoryReservation> MemoryReserveOrWait::reserve_or_wait(
    std::size_t size, std::int64_t net_memory_delta
) {
    // First, check whether the requested memory is immediately available.
    auto [res, _] = br_->reserve(mem_type_, size, AllowOverbooking::NO);
    if (res.size() == size) {
        co_return std::move(res);
    }

    // Use libcoro's queue to track completion of this reservation request.
    // The queue will have at most one item: the fulfilled memory reservation.
    coro::queue<MemoryReservation> request_queue{};

    // Enqueue a reservation request under the mutex.
    std::unique_lock lock(mutex_);
    reservation_requests_.insert(
        Request{
            .size = size,
            .net_memory_delta = net_memory_delta,
            .sequence_number = sequence_counter++,
            .queue = request_queue
        }
    );

    // If no periodic memory check task is running, start one.
    std::optional<coro::task<void>> previous_periodic_task;
    if (!periodic_task_running_) {
        // A previous periodic task may exist but is guaranteed to be either already
        // finished or about to finish. This can happen when the last request was
        // extracted and the task is in the process of exiting.
        //
        // We take ownership of that task here and await it below before proceeding,
        // ensuring that at most one periodic task is active at any time.
        previous_periodic_task = std::move(periodic_memory_check_task_);
        periodic_memory_check_task_ = executor_->spawn_joinable(periodic_memory_check());
        // Claim the slot until the task releases it.
        periodic_task_running_ = true;
    }
    lock.unlock();

    // If a previous periodic task existed, wait for it to fully exit before
    // continuing. The await must happen without holding the mutex, otherwise the
    // periodic task could deadlock while trying to acquire the same mutex.
    if (previous_periodic_task.has_value()) {
        co_await *previous_periodic_task;
    }

    // Suspend until our request is fulfilled.
    auto request = co_await request_queue.pop();
    RAPIDSMPF_EXPECTS(
        request.has_value(), "memory reservation failed", std::runtime_error
    );
    co_return std::move(*request);
}

coro::task<std::pair<MemoryReservation, std::size_t>>
MemoryReserveOrWait::reserve_or_wait_or_overbook(
    std::size_t size, std::int64_t net_memory_delta
) {
    auto ret = co_await reserve_or_wait(size, net_memory_delta);
    if (ret.size() < size) {
        co_return br_->reserve(mem_type_, size, AllowOverbooking::YES);
    }
    co_return {std::move(ret), 0};
}

coro::task<MemoryReservation> MemoryReserveOrWait::reserve_or_wait_or_fail(
    std::size_t size, std::int64_t net_memory_delta
) {
    auto ret = co_await reserve_or_wait(size, net_memory_delta);
    RAPIDSMPF_EXPECTS(
        ret.size() == size,
        "cannot reserve " + std::string{to_string(mem_type_)} + " memory ("
            + format_nbytes(size) + ")",
        rapidsmpf::reservation_error
    );
    co_return ret;
}

std::size_t MemoryReserveOrWait::size() const noexcept {
    std::lock_guard lock(mutex_);
    return reservation_requests_.size();
}

std::size_t MemoryReserveOrWait::periodic_memory_check_counter() const noexcept {
    return periodic_memory_check_counter_.load(std::memory_order_acquire);
}

std::shared_ptr<CoroThreadPoolExecutor> const&
MemoryReserveOrWait::executor() const noexcept {
    return executor_;
}

std::shared_ptr<BufferResource> const& MemoryReserveOrWait::br() const noexcept {
    return br_;
}

Duration MemoryReserveOrWait::timeout() const noexcept {
    return timeout_;
}

coro::task<void> MemoryReserveOrWait::periodic_memory_check() {
    // Helper that returns the memory available for new reservations, clamped so
    // negative values become zero.
    auto memory_available = [this]() -> std::size_t {
        std::int64_t const ret = br_->memory_available_for_reservation(mem_type_);
        return safe_cast<std::size_t>(std::max(ret, std::int64_t{0}));
    };

    // Helper that returns the subrange of reservation requests with size <= max_size.
    auto eligible_requests = [this](std::size_t max_size)
        -> std::ranges::subrange<std::set<Request>::const_iterator> {
        // Since `reservation_requests_` is sorted by ascending size,
        // upper_bound finds the first element with size > max_size.
        auto last = std::ranges::upper_bound(
            reservation_requests_, max_size, std::less<>{}, &Request::size
        );
        // The range [begin, last) contains all requests with size <= max_size.
        return {reservation_requests_.begin(), last};
    };

    // Helper that pushes a memory reservation into a request's queue **without**
    // waiting on the coroutine.
    auto push_into_queue =
        [this](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> void {
        auto err = executor_->spawn_detached(
            [](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> Actor {
                RAPIDSMPF_EXPECTS(
                    co_await queue.push(std::move(res))
                        == coro::queue_produce_result::produced,
                    "could not push memory reservation"
                );
            }(queue, std::move(res))
        );
        RAPIDSMPF_EXPECTS(err, "cannot spawn push-into-queue task");
    };

    // Helper that spills until `headroom` bytes are reservable, swallowing a spill
    // function that throws and logging it once per task. Returns false only when a
    // non-blocking spill found the spill lock held.
    bool spill_failure_logged = false;
    auto make_headroom = [&](std::size_t headroom, bool blocking) -> bool {
        // Only device memory is supported, see `spill_to_make_headroom()`.
        if (mem_type_ != MemoryType::DEVICE) {
            return true;
        }
        try {
            auto const target = safe_cast<std::int64_t>(headroom);
            if (blocking) {
                br_->spill_manager().spill_to_make_headroom(target);
                return true;
            }
            return br_->spill_manager().try_spill_to_make_headroom(target).has_value();
        } catch (...) {
            if (!std::exchange(spill_failure_logged, true)) {
                try {
                    logger_->warn(
                        "a spill function threw while making headroom for ",
                        format_nbytes(headroom),
                        ", falling back to the reservation timeout"
                    );
                } catch (...) {  // NOLINT(bugprone-empty-catch)
                }
            }
            return true;
        }
    };

    // RAII helper that releases `periodic_task_running_` when this task exits without
    // reaching one of the `co_return` paths below, such as on an exception. Those paths
    // release the flag under the same lock acquisition that observes the empty request
    // set, and dismiss the guard.
    struct RunningFlagGuard {
        MemoryReserveOrWait* self;

        ~RunningFlagGuard() {
            if (self != nullptr) {
                std::lock_guard lock(self->mutex_);
                self->periodic_task_running_ = false;
            }
        }

        void dismiss() noexcept {
            self = nullptr;
        }
    };

    RunningFlagGuard running_flag_guard{.self = this};

    while (true) {
        auto last_reservation_success = Clock::now();
        // Spilling is attempted once per timeout window, re-armed by each served
        // request.
        bool spill_attempted = false;
        while (true) {
            // Exit if no more pending requests remain.
            {
                std::unique_lock lock(mutex_);
                if (reservation_requests_.empty()) {
                    periodic_task_running_ = false;
                    running_flag_guard.dismiss();
                    co_return;
                }
            }
            periodic_memory_check_counter_.fetch_add(1, std::memory_order_acq_rel);
            co_await executor_->yield();
            if (Clock::now() - last_reservation_success > timeout_) {
                // This is the only way out of the while-loop that doesn't shutdown
                // the periodic memory check.
                break;
            }
            auto const max_size = memory_available();

            // Find the request with the smallest net_memory_delta that fits
            // into the currently available memory.
            std::unique_lock lock(mutex_);
            auto eligibles = eligible_requests(max_size);
            if (eligibles.empty()) {
                // Nothing fits. Spill enough to admit the request with the smallest
                // net_memory_delta, then re-poll.
                if (!spill_attempted && !reservation_requests_.empty()) {
                    auto const target = std::ranges::min_element(
                        reservation_requests_, std::less<>{}, &Request::net_memory_delta
                    );
                    auto const headroom = target->size;
                    // Spill without holding the mutex, the spill functions take the
                    // buffer resource's lock.
                    lock.unlock();
                    spill_attempted = make_headroom(headroom, /* blocking = */ false);
                }
                continue;  // No eligible requests.
            }

            auto it = std::ranges::min_element(
                eligibles, std::less<>{}, &Request::net_memory_delta
            );

            // Try to reserve memory for the selected request.
            auto [res, _] = br_->reserve(mem_type_, it->size, AllowOverbooking::NO);
            if (res.size() == 0) {
                continue;  // Memory is no longer available.
            }

            // Extract the selected request and push the reservation into its queue.
            Request request = reservation_requests_.extract(it).value();
            lock.unlock();
            push_into_queue(request.queue, std::move(res));
            last_reservation_success = Clock::now();
            spill_attempted = false;
        }

        // Reaching this point means we hit the timeout. We force progress by selecting
        // among the smallest pending requests, preferring the one with the smallest
        // net_memory_delta.
        std::unique_lock lock(mutex_);
        if (reservation_requests_.empty()) {
            periodic_task_running_ = false;
            running_flag_guard.dismiss();
            co_return;
        }

        // The set is sorted by size (ascending). First, find the smallest size.
        auto first = reservation_requests_.begin();
        auto const smallest_size = first->size;

        // Consider all requests with that size.
        auto same_size_end = std::ranges::upper_bound(
            reservation_requests_, smallest_size, std::less<>{}, &Request::size
        );

        // Among the smallest requests, pick the one with the smallest
        // net_memory_delta. If multiple requests tie, we pick the oldest one,
        // since the set is ordered by size and then sequence_number (ascending).
        auto it = std::ranges::min_element(
            std::ranges::subrange(first, same_size_end),
            std::less<>{},
            &Request::net_memory_delta
        );

        Request request = reservation_requests_.extract(it).value();
        lock.unlock();

        // Last chance before handing back a zero-size reservation.
        make_headroom(request.size, /* blocking = */ true);

        // Reserve memory and accept a zero-size result if it does not fit into the
        // currently available memory.
        auto [res, _] = br_->reserve(mem_type_, request.size, AllowOverbooking::NO);
        push_into_queue(request.queue, std::move(res));
    }
}

coro::task<MemoryReservation> reserve_memory(
    std::shared_ptr<Context> ctx,
    std::size_t size,
    std::int64_t net_memory_delta,
    MemoryType mem_type,
    std::optional<AllowOverbooking> allow_overbooking
) {
    // If allow_overbooking is not specified, get it from the configuration options.
    if (!allow_overbooking.has_value()) {
        bool const allow_overbook_default =
            ctx->options().get<bool>("allow_overbooking_by_default", parse_string<bool>);
        allow_overbooking =
            allow_overbook_default ? AllowOverbooking::YES : AllowOverbooking::NO;
    }

    // Reserve memory based on the overbooking policy.
    if (allow_overbooking.value() == AllowOverbooking::YES) {
        auto [res, _] = co_await ctx->memory(mem_type)->reserve_or_wait_or_overbook(
            size, net_memory_delta
        );
        co_return std::move(res);
    } else {
        co_return co_await ctx->memory(mem_type)->reserve_or_wait_or_fail(
            size, net_memory_delta
        );
    }
}

}  // namespace rapidsmpf::streaming
