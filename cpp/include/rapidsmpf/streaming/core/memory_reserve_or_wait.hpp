/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils.hpp>

#include <coro/task.hpp>

namespace rapidsmpf::streaming {


/**
 * @brief Asynchronous coordinator for memory reservation requests.
 *
 * `MemoryReserveOrWait` provides a coroutine-based mechanism for reserving
 * memory with backpressure. Callers submit reservation requests via
 * `reserve_or_wait()`, which suspends until enough memory is available.
 */
class MemoryReserveOrWait {
  public:
    /**
     * @brief Constructs a `MemoryReserveOrWait` instance.
     *
     * @param mem_type The memory type for which reservations are requested.
     * @param ctx Streaming context.
     * @param timeout Optional timeout duration. This timeout applies to how long pending
     * requests may wait without making progress. If the timeout expires, a
     * `reserve_or_wait()` returns even if no memory became available. If not explicitly
     * provided, the timeout is read from the option key `"memory_reserve_timeout_ms"`,
     * which defaults to 100 ms.
     */
    MemoryReserveOrWait(
        MemoryType mem_type,
        std::shared_ptr<Context> ctx,
        std::optional<Duration> timeout = std::nullopt
    );

    ~MemoryReserveOrWait() noexcept;

    /**
     * @brief Shuts down all pending memory reservation requests.
     *
     * @return A coroutine that completes only after all pending requests have been
     * cancelled and the periodic memory check coroutine has exited.
     */
    Node shutdown();

    /**
     * @brief Attempts to reserve memory or waits until the reservation can be satisfied.
     *
     * This coroutine submits a memory reservation request and then suspends until
     * either sufficient memory becomes available or no progress is made within the
     * configured timeout.
     *
     * If the timeout expires before the request can be fulfilled, an empty
     * `MemoryReservation` is returned.
     *
     * @param size Number of bytes to reserve.
     * @param future_release_potential Estimated number of bytes the requester may release
     * in the future, used as a heuristic when selecting which eligible request to satisfy
     * first.
     * @return A `MemoryReservation` representing the allocated memory, or an empty
     * reservation if the timeout expires.
     *
     * @throw std::runtime_error If shutdown occurs before the request can be processed.
     */
    coro::task<MemoryReservation> reserve_or_wait(
        std::size_t size, std::size_t future_release_potential
    );

    /**
     * @brief Returns the number of pending memory reservation requests.
     *
     * It may change concurrently as requests are added or fulfilled.
     *
     * @return The number of outstanding reservation requests.
     */
    [[nodiscard]] std::size_t size() const;

    /**
     * @brief Returns the number of iterations performed by the `periodic_memory_check()`.
     *
     * This counter is incremented once per loop iteration inside
     * `periodic_memory_check()`, and can be useful for diagnostics or testing.
     *
     * @return The total number of memory-check iterations executed so far.
     */
    [[nodiscard]] std::size_t periodic_memory_check_counter() const;

  private:
    /**
     * @brief Represents a single memory reservation request.
     *
     * A `ResReq` is inserted into a sorted container and processed by
     * `periodic_memory_check()`. Each request describes the amount of memory
     * needed, an estimate of how much memory may be released in the future, and
     * its submission order. A reference to the requester's queue is used to
     * deliver the resulting `MemoryReservation` once the request is fulfilled.
     *
     * The ordering of `ResReq` instances is defined by `operator<`, which sorts
     * primarily by `size` (ascending).
     */
    struct ResReq {
        /// @brief The number of bytes requested.
        std::size_t size;

        /// @brief  Estimated number of bytes expected to be released in the future.
        std::size_t future_release_potential;

        /// @brief Monotonically increasing identifier used to preserve submission order.
        std::uint64_t sequence_number;

        /// @brief Queue into which a reservation is pushed once the request is satisfied.
        coro::queue<MemoryReservation>& queue;

        /// @brief Lexicographic ordering.
        friend bool operator<(ResReq const& a, ResReq const& b) {
            return std::tie(a.size, a.future_release_potential, a.sequence_number)
                   < std::tie(b.size, b.future_release_potential, b.sequence_number);
        }
    };

    /**
     * @brief Periodically processes pending memory reservation requests.
     *
     * This coroutine drives the asynchronous mechanism of `MemoryReserveOrWait`.
     * It repeatedly:
     *  - Queries the currently available memory for the configured memory type.
     *  - Identifies all pending reservation requests whose `size` fits within the
     *    available memory.
     *  - Among those, selects the request with the largest `future_release_potential`.
     *  - Fulfills the request by creating a `MemoryReservation` and pushing it into
     *    the requester's queue.
     *
     * Shutdown and lifetime coordination
     * ----------------------------------
     * To avoid undefined behavior, the destructor must not return until
     * `periodic_memory_check()` has fully exited. This is coordinated using
     * `shutdown_event_`, which acts as the synchronization point between the
     * coroutine and object teardown:
     *  - `periodic_memory_check_with_error_handling()` sets `shutdown_event_`
     *    once `periodic_memory_check()` has finished.
     *  - The destructor (and `shutdown()`) waits on this event before returning,
     *    ensuring the coroutine has terminated.
     *
     * @return A coroutine that completes only once shutdown has been requested and all
     * in-flight work has finished.
     */
    coro::task<void> periodic_memory_check();

    /**
     * @brief Runs `periodic_memory_check()` with exception handling and coordinates
     * shutdown via `shutdown_event_`.
     */
    coro::task<void> periodic_memory_check_with_error_handling();

    mutable std::mutex mutex_;
    std::uint64_t sequence_counter{0};
    MemoryType const mem_type_;
    std::shared_ptr<Context> ctx_;
    Duration const timeout_;
    std::set<ResReq> reservation_requests_;
    std::atomic<std::uint64_t> periodic_memory_check_counter_{0};
    coro::event shutdown_event_{/* initially_set = */ true};
};

}  // namespace rapidsmpf::streaming
