/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>

#include <coro/sync_wait.hpp>

namespace rapidsmpf::streaming {
namespace {
constexpr bool no_overbooking = false;
}

MemoryReserveOrWait::MemoryReserveOrWait(
    MemoryType mem_type, std::shared_ptr<Context> ctx, std::optional<Duration> timeout
)
    : mem_type_{mem_type},
      ctx_{std::move(ctx)},
      timeout_{
          // Use timeout if it is set, otherwise read the Context.options().
          timeout.has_value()
              ? *timeout
              : ctx_->options().get<Duration>(
                    "memory_reserve_timeout_ms", [](std::string const& s) {
                        return s.empty() ? std::chrono::milliseconds{100}
                                         : std::chrono::milliseconds{std::stoi(s)};
                    }
                )
      } {}

MemoryReserveOrWait::~MemoryReserveOrWait() noexcept {
    coro::sync_wait(shutdown());
}

Node MemoryReserveOrWait::shutdown() {
    // Extract the reservation request and shutdown their queues.
    std::unique_lock lock(mutex_);
    auto reservation_requests = std::move(reservation_requests_);
    lock.unlock();
    if (!reservation_requests.empty()) {
        std::vector<Node> nodes;
        for (ResReq request : reservation_requests) {
            nodes.push_back(request.queue.shutdown_drain(ctx_->executor()));
        }
        coro_results(co_await coro::when_all(std::move(nodes)));
    }
    co_await shutdown_event_;
}

coro::task<MemoryReservation> MemoryReserveOrWait::reserve_or_wait(
    std::size_t size, std::size_t future_release_potential
) {
    // First, check whether the requested memory is immediately available.
    auto [res, _] = ctx_->br()->reserve(mem_type_, size, no_overbooking);
    if (res.size() > 0) {
        co_return std::move(res);
    }

    // Use libcoro's queue to track completion of this reservation request.
    // The queue will have at most one item: the fulfilled memory reservation.
    coro::queue<MemoryReservation> request_queue{};

    // Since the memory is not immediately available, insert a reservation request
    // while holding the mutex.
    std::unique_lock lock(mutex_);
    bool const spawn_periodic_memory_check = reservation_requests_.empty();
    reservation_requests_.insert(
        ResReq{
            .size = size,
            .future_release_potential = future_release_potential,
            .sequence_number = sequence_counter++,
            .queue = request_queue
        }
    );
    lock.unlock();

    // If this is the first pending request, start the periodic memory check task.
    if (spawn_periodic_memory_check) {
        shutdown_event_.reset();
        RAPIDSMPF_EXPECTS(
            ctx_->executor()->spawn(periodic_memory_check_with_error_handling()),
            "cannot spawn task"
        );
    }

    // Suspend until our request is fulfilled.
    auto request = co_await request_queue.pop();
    RAPIDSMPF_EXPECTS(
        request.has_value(), "memory reservation failed", std::runtime_error
    );
    co_return std::move(*request);
}

std::size_t MemoryReserveOrWait::size() const {
    std::lock_guard lock(mutex_);
    return reservation_requests_.size();
}

std::size_t MemoryReserveOrWait::periodic_memory_check_counter() const {
    return periodic_memory_check_counter_.load(std::memory_order_acquire);
}

coro::task<void> MemoryReserveOrWait::periodic_memory_check() {
    BufferResource* br = ctx_->br();

    // Helper that returns available memory, clamped so negative values become zero.
    auto memory_available = [f = br->memory_available(mem_type_)]() -> std::size_t {
        auto const ret = f();
        return ret < 0 ? 0 : static_cast<std::size_t>(ret);
    };

    // Helper that returns the subrange of reservation requests with size <= max_size.
    auto eligible_requests = [this](std::size_t max_size)
        -> std::ranges::subrange<std::set<ResReq>::const_iterator> {
        // Since `reservation_requests_` is sorted by ascending size,
        // upper_bound finds the first element with size > max_size.
        auto last = std::ranges::upper_bound(
            reservation_requests_, max_size, std::less<>{}, &ResReq::size
        );
        // The range [begin, last) contains all requests with size <= max_size.
        return {reservation_requests_.begin(), last};
    };

    // Helper that pushes a memory reservation into a request's queue **without**
    // waiting on the coroutine.
    auto push_into_queue = [this](ResReq& request, MemoryReservation res) -> void {
        auto err = ctx_->executor()->spawn(
            [](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> Node {
                RAPIDSMPF_EXPECTS(
                    co_await queue.push(std::move(res))
                        == coro::queue_produce_result::produced,
                    "could not push memory reservation"
                );
            }(request.queue, std::move(res))
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
            co_await ctx_->executor()->yield();
            if (Clock::now() - last_reservation_success > timeout_) {
                // This is the only way out of the while-loop that doesn't shutdown
                // the periodic memory check.
                break;
            }
            auto const max_size = memory_available();

            // Find the request with the greatest future_release_potential that fits
            // into the currently available memory.
            std::unique_lock lock(mutex_);
            auto eligibles = eligible_requests(max_size);
            if (eligibles.begin() == eligibles.end()) {
                continue;  // No eligible requests.
            }
            auto it = std::ranges::max_element(
                eligibles, std::less<>{}, &ResReq::future_release_potential
            );

            // Try to reserve memory for the selected request.
            auto [res, _] = ctx_->br()->reserve(mem_type_, it->size, no_overbooking);
            if (res.size() == 0) {
                continue;  // Memory is no longer available.
            }

            // Extract the selected request and push the reservation into its queue.
            ResReq request = reservation_requests_.extract(it).value();
            lock.unlock();
            push_into_queue(request, std::move(res));
            last_reservation_success = Clock::now();
        }

        // Reaching this point means we hit the timeout. Let's extract the smallest
        // request, even if it does not fit in the available memory.
        std::unique_lock lock(mutex_);
        if (reservation_requests_.empty()) {
            co_return;
        }
        // The set is already sorted by size (ascending) so we pick the beginning.
        ResReq request =
            reservation_requests_.extract(reservation_requests_.begin()).value();
        lock.unlock();

        // Reserve memory and accept a zero-size result if it does not fit into the
        // currently available memory.
        auto [res, _] = ctx_->br()->reserve(mem_type_, request.size, no_overbooking);
        push_into_queue(request, std::move(res));
    }
}

coro::task<void> MemoryReserveOrWait::periodic_memory_check_with_error_handling() {
    try {
        co_await periodic_memory_check();
    } catch (std::exception const& e) {
        std::cerr << "periodic_memory_check(): unhandled exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "periodic_memory_check(): unhandled exception\n";
    }
    // Ensure we always sets `shutdown_event_` and shutdowns gracefully.
    shutdown_event_.set();
    co_await shutdown();
}

}  // namespace rapidsmpf::streaming
