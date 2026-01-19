/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>

#include <coro/sync_wait.hpp>

namespace rapidsmpf::streaming {

MemoryReserveOrWait::MemoryReserveOrWait(
    config::Options options,
    MemoryType mem_type,
    std::shared_ptr<CoroThreadPoolExecutor> executor,
    std::shared_ptr<BufferResource> br
)
    : mem_type_{mem_type},
      executor_{std::move(executor)},
      br_{std::move(br)},
      timeout_{
          options.get<Duration>("memory_reserve_timeout_ms", [](std::string const& s) {
              return s.empty() ? std::chrono::milliseconds{100}
                               : std::chrono::milliseconds{std::stoi(s)};
          })
      } {
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");
}

MemoryReserveOrWait::~MemoryReserveOrWait() noexcept {
    coro::sync_wait(shutdown());
}

Node MemoryReserveOrWait::shutdown() {
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
        std::vector<Node> nodes;
        for (Request const& request : reservation_requests) {
            nodes.push_back(request.queue.shutdown());
        }
        coro_results(co_await coro::when_all(std::move(nodes)));
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
    bool const spawn_periodic_memory_check = reservation_requests_.empty();
    reservation_requests_.insert(
        Request{
            .size = size,
            .net_memory_delta = net_memory_delta,
            .sequence_number = sequence_counter++,
            .queue = request_queue
        }
    );

    // If this is the first pending request, start the periodic memory check task.
    std::optional<coro::task<void>> previous_periodic_task;
    if (spawn_periodic_memory_check) {
        // A previous periodic task may exist but is guaranteed to be either already
        // finished or about to finish. This can happen when the last request was
        // extracted and the task is in the process of exiting.
        //
        // We take ownership of that task here and await it below before proceeding,
        // ensuring that at most one periodic task is active at any time.
        previous_periodic_task = std::move(periodic_memory_check_task_);
        periodic_memory_check_task_ = executor_->spawn_joinable(periodic_memory_check());
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
        std::overflow_error
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

std::shared_ptr<CoroThreadPoolExecutor> MemoryReserveOrWait::executor() const noexcept {
    return executor_;
}

std::shared_ptr<BufferResource> MemoryReserveOrWait::br() const noexcept {
    return br_;
}

Duration MemoryReserveOrWait::timeout() const noexcept {
    return timeout_;
}

coro::task<void> MemoryReserveOrWait::periodic_memory_check() {
    // Helper that returns available memory, clamped so negative values become zero.
    auto memory_available = [f = br_->memory_available(mem_type_)]() -> std::size_t {
        std::int64_t const ret = f();
        return static_cast<std::size_t>(std::max(ret, std::int64_t{0}));
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
            [](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> Node {
                RAPIDSMPF_EXPECTS(
                    co_await queue.push(std::move(res))
                        == coro::queue_produce_result::produced,
                    "could not push memory reservation"
                );
            }(queue, std::move(res))
        );
        RAPIDSMPF_EXPECTS(err, "cannot spawn push-into-queue task");
    };

    while (true) {
        auto last_reservation_success = Clock::now();
        while (true) {
            // Exit if no more pending requests remain.
            {
                std::unique_lock lock(mutex_);
                if (reservation_requests_.empty()) {
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
        }

        // Reaching this point means we hit the timeout. We force progress by selecting
        // among the smallest pending requests, preferring the one with the smallest
        // net_memory_delta.
        std::unique_lock lock(mutex_);
        if (reservation_requests_.empty()) {
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

        // Reserve memory and accept a zero-size result if it does not fit into the
        // currently available memory.
        auto [res, _] = br_->reserve(mem_type_, request.size, AllowOverbooking::NO);
        push_into_queue(request.queue, std::move(res));
    }
}

}  // namespace rapidsmpf::streaming
