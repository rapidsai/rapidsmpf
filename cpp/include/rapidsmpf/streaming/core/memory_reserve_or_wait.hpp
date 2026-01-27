/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <set>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/streaming/core/coro_executor.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <coro/task.hpp>

namespace rapidsmpf::streaming {

// Forward declaration
class Context;

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
     * If no reservation request can be satisfied within the timeout specified by
     * the `"memory_reserve_timeout"` key in @p options, the coroutine forces
     * progress by selecting the smallest pending request and attempting to reserve
     * memory for it. This attempt may result in an empty reservation if the request
     * still cannot be satisfied.
     *
     * @param options Configuration options.
     * @param mem_type The memory type for which reservations are requested.
     * @param executor Shared pointer to a coroutine executor.
     * @param br Buffer resource for memory allocation.*
     */
    MemoryReserveOrWait(
        config::Options options,
        MemoryType mem_type,
        std::shared_ptr<CoroThreadPoolExecutor> executor,
        std::shared_ptr<BufferResource> br
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
     * either sufficient memory becomes available or no reservation request
     * (including other pending requests) makes progress within the configured
     * timeout.
     *
     * The timeout does not apply specifically to this request. Instead, it is used
     * as a global progress guarantee: if no pending reservation request can be
     * satisfied within the timeout, `MemoryReserveOrWait` forces progress by
     * selecting the smallest pending request and attempting to reserve memory for
     * it. The forced reservation attempt may result in an empty `MemoryReservation`
     * if the selected request still cannot be satisfied.
     *
     * When multiple reservation requests are eligible, `MemoryReserveOrWait` uses
     * @p net_memory_delta as a heuristic to prefer requests that are expected to
     * reduce memory pressure sooner. The value represents the estimated net change
     * in memory usage after the reservation has been allocated and the dependent
     * operation completes (that is, the memory impact after both allocating @p size
     * and finishing the work that consumes the reservation):
     *   - > 0: expected net increase in memory usage
     *   - = 0: memory-neutral
     *   - < 0: expected net decrease in memory usage
     *
     * Smaller values have higher priority. Examples:
     *   - Reading data from disk into memory typically has a positive
     *     @p net_memory_delta (memory usage increases).
     *   - A row-wise transformation that retains input and output typically has a
     *     net delta near zero (memory-neutral).
     *   - Writing data to disk or a reduction that frees inputs typically has a
     *     negative @p net_memory_delta (memory usage decreases).
     *
     * @param size Number of bytes to reserve.
     * @param net_memory_delta Estimated net change in memory usage after the
     * reservation is allocated and the dependent operation completes. Smaller
     * values have higher priority.
     * @return A `MemoryReservation` representing the allocated memory, or an empty
     * reservation if progress could not be made.
     *
     * @throws std::runtime_error If shutdown occurs before the request can be
     * processed.
     */
    coro::task<MemoryReservation> reserve_or_wait(
        std::size_t size, std::int64_t net_memory_delta
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
     * @param net_memory_delta Heuristic used to prioritize eligible requests. See
     * `reserve_or_wait()` for details and semantics.
     * @return A pair consisting of:
     *   - A `MemoryReservation` representing the allocated memory.
     *   - The number of bytes by which the reservation overbooked the available
     *     memory. This value is zero if no overbooking occurred.
     *
     * @throws std::runtime_error If shutdown occurs before the request can be
     * processed.
     *
     * @see reserve_or_wait()
     */
    coro::task<std::pair<MemoryReservation, std::size_t>> reserve_or_wait_or_overbook(
        std::size_t size, std::int64_t net_memory_delta
    );

    /**
     * @brief Variant of `reserve_or_wait()` that fails if no progress is possible.
     *
     * This coroutine behaves identically to `reserve_or_wait()` with respect to
     * request submission, waiting, and progress guarantees until the progress
     * timeout expires.
     *
     * If no reservation request can be satisfied before the timeout, this method
     * fails instead of forcing progress. Overbooking is not allowed, and no memory
     * reservation is made.
     *
     * @param size Number of bytes to reserve.
     * @param net_memory_delta Heuristic used to prioritize eligible requests. See
     * `reserve_or_wait()` for details and semantics.
     * @return A `MemoryReservation` representing the allocated memory.
     *
     * @throws std::overflow_error If no progress is possible within the timeout.
     * @throws std::runtime_error If shutdown occurs before the request can be
     * processed.
     *
     * @see reserve_or_wait()
     */
    coro::task<MemoryReservation> reserve_or_wait_or_fail(
        std::size_t size, std::int64_t net_memory_delta
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

    /**
     * @brief Get the coroutine executor used by this instance.
     *
     * @return Shared pointer to the coroutine executor.
     */
    [[nodiscard]] std::shared_ptr<CoroThreadPoolExecutor> executor() const noexcept;

    /**
     * @brief Get the buffer resource used for memory reservations.
     *
     * @return Shared pointer to the buffer resource.
     */
    [[nodiscard]] std::shared_ptr<BufferResource> br() const noexcept;

    /**
     * @brief Get the configured progress timeout.
     *
     * @return The progress timeout duration.
     */
    [[nodiscard]] Duration timeout() const noexcept;

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

        /// @brief Estimated net change in memory usage after the reservation is allocated
        /// and the dependent operation completes. Smaller values have higher priority.
        std::int64_t net_memory_delta;

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
     *  - Among those, selects the request with the smallest `net_memory_delta`.
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
    std::shared_ptr<CoroThreadPoolExecutor> executor_;
    std::shared_ptr<BufferResource> br_;
    Duration const timeout_;
    std::set<Request> reservation_requests_;
    std::atomic<std::uint64_t> periodic_memory_check_counter_{0};
    std::optional<coro::task<void>> periodic_memory_check_task_;
};

/**
 * @brief Reserve memory using the context memory reservation mechanism.
 *
 * Submits a memory reservation request for the configured memory type and suspends
 * until the request is satisfied. If no pending reservation request can be satisfied
 * within the configured `"memory_reserve_timeout"`, the behavior depends on
 * @p allow_overbooking.
 *
 * This is a convenience helper that returns only the `MemoryReservation`. If more
 * control is required, for example inspecting the amount of overbooking, callers
 * should use `MemoryReserveOrWait` directly, such as
 * `ctx.memory(MemoryType::DEVICE).reserve_or_wait_or_overbook(size, net_memory_delta)`.
 *
 * Priority and progress semantics are identical to
 * `MemoryReserveOrWait::reserve_or_wait()`. In particular, @p net_memory_delta is used as
 * a heuristic to prefer eligible requests that are expected to reduce memory pressure
 * sooner. Smaller values have higher priority.
 *
 * @param ctx Node context used to obtain the memory reservation handle.
 * @param size Number of bytes to reserve.
 * @param net_memory_delta Estimated net change in memory usage after the reservation is
 * allocated and the dependent operation completes. Smaller values have higher priority.
 * @param mem_type Memory type for which to reserve memory.
 * @param allow_overbooking Controls the behavior when no progress is possible within the
 * configured timeout:
 * - If set to `AllowOverbooking::YES`, the call may overbook memory when forcing
 *   progress.
 * - If set to `AllowOverbooking::NO`, the call fails if no progress is possible.
 * - If not provided, the default behavior is determined by the configuration option
 *   `"allow_overbooking_by_default"`.
 * @return The allocated memory reservation.
 *
 * @throws std::runtime_error If shutdown occurs before the request can be processed.
 * @throws std::overflow_error If no progress is possible within the timeout and
 * `allow_overbooking` resolves to `AllowOverbooking::NO`.
 *
 * @code{.cpp}
 * // Reserve memory inside a node:
 * auto res = co_await reserve_memory(
 *     ctx,
 *     1024,
 *     0,  // net_memory_delta
 *     MemoryType::DEVICE,
 *     AllowOverbooking::YES
 * );
 * EXPECT_EQ(res.size(), 1024);
 *
 * // Disable overbooking and fail if no progress is possible:
 * auto res2 = co_await reserve_memory(
 *     ctx,
 *     2048,
 *     0,  // net_memory_delta
 *     MemoryType::DEVICE,
 *     AllowOverbooking::NO
 * );
 * @endcode
 *
 * @see MemoryReserveOrWait::reserve_or_wait()
 * @see MemoryReserveOrWait::reserve_or_wait_or_overbook()
 * @see MemoryReserveOrWait::reserve_or_wait_or_fail()
 */
coro::task<MemoryReservation> reserve_memory(
    std::shared_ptr<Context> ctx,
    std::size_t size,
    std::int64_t net_memory_delta,
    MemoryType mem_type = MemoryType::DEVICE,
    std::optional<AllowOverbooking> allow_overbooking = std::nullopt
);

}  // namespace rapidsmpf::streaming
