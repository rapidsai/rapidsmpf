/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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
 * `reserve_or_wait()`, which suspends until enough memory is available or the
 * request times out.
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
     * cancelled and the periodic memory check task has exited.
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
     * @throws std::runtime_error If shutdown occurs before the request can be processed.
     */
    coro::task<MemoryReservation> reserve_or_wait(
        std::size_t size, std::size_t future_release_potential
    );

    /**
     * @brief Attempts to reserve memory, waits, or overbooks on timeout.
     *
     * This coroutine submits a memory reservation request and suspends until
     * either sufficient memory becomes available or no progress is made within the
     * configured timeout.
     *
     * If the request cannot be satisfied within the timeout, the function attempts
     * to reserve the requested memory by allowing overbooking. This guarantees
     * forward progress at the cost of potentially exceeding the configured memory
     * limits.
     *
     * @param size Number of bytes to reserve.
     * @param future_release_potential Estimated number of bytes the requester may release
     * in the future, used as a heuristic when selecting which eligible request to satisfy
     * first.
     * @return A pair consisting of:
     *   - A `MemoryReservation` representing the allocated memory.
     *   - The number of bytes by which the reservation overbooked the available memory.
     *     This value is zero if no overbooking occurred.
     *
     * @throws std::runtime_error If shutdown occurs before the request can be processed.
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
    [[nodiscard]] std::size_t size() const;

    /**
     * @brief Returns the number of iterations performed by `periodic_memory_check()`.
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
     * lexicographically by `(size, future_release_potential, sequence_number)`.
     */
    struct ResReq {
        /// @brief The number of bytes requested.
        std::size_t size;

        /// @brief Estimated number of bytes expected to be released in the future.
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
    std::set<ResReq> reservation_requests_;
    std::atomic<std::uint64_t> periodic_memory_check_counter_{0};
    std::optional<coro::task<void>> periodic_memory_check_task_;
};

}  // namespace rapidsmpf::streaming
