/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
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
 * `reserve_or_wait()`, which suspends until enough memory is available or
 * progress must be forced.
 */
class MemoryReserveOrWait {
  public:
    /**
     * @brief Constructs a `MemoryReserveOrWait` instance.
     *
     * If no reservation request can be satisfied within @p timeout, the coroutine
     * forces progress by selecting the smallest pending request and attempting to
     * reserve memory for it. This attempt may result in an empty reservation if the
     * request still cannot be satisfied.
     *
     * @param mem_type The memory type for which reservations are requested.
     * @param ctx Streaming context.
     * @param timeout Timeout duration. If not explicitly provided, the timeout is read
     * from the option key `"memory_reserve_timeout_ms"`, which defaults to 100 ms.
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
     * cancelled and the periodic memory check task has exited.
     */
    Node shutdown();

    /**
     * @brief Attempts to reserve memory or waits until progress can be made.
     *
     * This coroutine submits a memory reservation request and then suspends until
     * either sufficient memory becomes available or no reservation request (including
     * other pending requests) makes progress within the configured timeout.
     *
     * The timeout does not apply specifically to this request. Instead, it is used as
     * a global progress guarantee: if no pending reservation request can be satisfied
     * within timeout, `MemoryReserveOrWait` forces progress by selecting the smallest
     * pending request and attempting to reserve memory for it. The forced reservation
     * attempt may result in an empty `MemoryReservation` if the selected request still
     * cannot be satisfied.
     *
     * When multiple reservation requests are eligible, `MemoryReserveOrWait` uses
     * @p future_release_potential as a heuristic to prefer requests that are expected
     * to free memory sooner. Operations that do not free memory, for example reading
     * data from disk into memory, should use a value of zero. Operations that are
     * expected to reduce memory usage, for example a reduction such as a sum, should
     * use a value corresponding to the amount of input data that will be released
     * once the operation completes.
     *
     * @param size Number of bytes to reserve.
     * @param future_release_potential Estimated number of bytes the requester may
     * release in the future.
     * @return A `MemoryReservation` representing the allocated memory, or an empty
     * reservation if progress could not be made.
     *
     * @throws std::runtime_error If shutdown occurs before the request can be processed.
     */
    coro::task<MemoryReservation> reserve_or_wait(
        std::size_t size, std::size_t future_release_potential
    );

    /**
     * @brief Variant of `reserve_or_wait()` that allows overbooking on timeout.
     *
     * This coroutine behaves identically to `reserve_or_wait()` with respect to
     * request submission, waiting, and progress guarantees. The only difference is
     * the behavior when the progress timeout expires.
     *
     * If no reservation request can be satisfied before the timeout, this method
     * attempts to reserve the requested memory by allowing overbooking. This
     * guarantees forward progress, but may exceed the configured memory limits.
     *
     * @param size Number of bytes to reserve.
     * @param future_release_potential Estimated number of bytes the requester may
     * release in the future.
     * @return A pair consisting of:
     *   - A `MemoryReservation` representing the allocated memory.
     *   - The number of bytes by which the reservation overbooked the available
     *     memory. This value is zero if no overbooking occurred.
     *
     * @throws std::runtime_error If shutdown occurs before the request can be processed.
     *
     * @see reserve_or_wait()
     */
    coro::task<std::pair<MemoryReservation, std::size_t>> reserve_or_wait_or_overbook(
        std::size_t size, std::size_t future_release_potential
    );

    /**
     * @brief Returns the number of pending memory reservation requests.
     *
     * It may change concurrently as requests are added or fulfilled.
     *
     * @return The number of outstanding reservation requests.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Returns the number of iterations performed by `periodic_memory_check()`.
     *
     * This counter is incremented once per loop iteration inside
     * `periodic_memory_check()`, and can be useful for diagnostics or testing.
     *
     * @return The total number of memory-check iterations executed so far.
     */
    [[nodiscard]] std::size_t periodic_memory_check_counter() const noexcept;

  private:
    /**
     * @brief Represents a single memory reservation request.
     *
     * A `Request` is inserted into a sorted container and processed by
     * `periodic_memory_check()`.
     */
    struct Request {
        /// @brief The number of bytes requested.
        std::size_t size;

        /// @brief Estimated number of bytes expected to be released in the future.
        std::size_t future_release_potential;

        /// @brief Monotonically increasing identifier used to preserve submission order.
        std::uint64_t sequence_number;

        /// @brief Queue into which a reservation is pushed once the request is satisfied.
        coro::queue<MemoryReservation>& queue;

        /// @brief Ordering by `size` and `sequence_number` (ascending).
        friend bool operator<(Request const& a, Request const& b) {
            return std::tie(a.size, a.sequence_number)
                   < std::tie(b.size, b.sequence_number);
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
     * If no reservation request can be satisfied for longer than `timeout_`, the
     * coroutine forces progress by selecting the smallest pending request and
     * attempting a reservation for it. This may produce an empty reservation if the
     * request still cannot be satisfied.
     *
     * Shutdown and lifetime coordination
     * ----------------------------------
     * A periodic memory check task is spawned on demand when the first pending
     * request is enqueued, and it exits once all requests have been extracted.
     *
     * The task is spawned as a joinable coroutine. `shutdown()` and the destructor
     * await the joinable task (if present) to ensure `periodic_memory_check()` has
     * fully exited before object teardown. This avoids dangling references to
     * members accessed by the coroutine.
     *
     * @return A coroutine that completes only once all pending requests have been
     * extracted and all in-flight work has finished.
     */
    coro::task<void> periodic_memory_check();

    mutable std::mutex mutex_;
    std::uint64_t sequence_counter{0};
    MemoryType const mem_type_;
    std::shared_ptr<Context> ctx_;
    Duration const timeout_;
    std::set<Request> reservation_requests_;
    std::atomic<std::uint64_t> periodic_memory_check_counter_{0};
    std::optional<coro::task<void>> periodic_memory_check_task_;
};

}  // namespace rapidsmpf::streaming
