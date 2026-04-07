/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <coro/event.hpp>
#include <coro/task.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Point-to-point halo exchange between adjacent ranks.
 *
 * Exchanges boundary data between neighboring ranks in a linear rank topology.
 * Each call to `exchange()` performs one round of bidirectional neighbor exchange:
 *
 * - Rank k sends `send_right` to rank k+1 and receives from rank k-1 as `from_left`.
 * - Rank k sends `send_left` to rank k-1 and receives from rank k+1 as `from_right`.
 *
 * @warning Calls to `exchange()` on one instance must be **sequential**: the
 * next call may only be issued after the previous coroutine completes.
 * Concurrent calls will corrupt per-round receive state.
 *
 * Multiple sequential calls on the same instance represent successive rounds of
 * halo propagation (for multi-hop window coverage when the rolling window period
 * exceeds a single rank's data range).
 *
 * ### Rolling window use case
 *
 * After a global sort, rank k owns a contiguous index interval. The rolling actor
 * buffers all local chunks and then calls `exchange()` in a loop until every rank
 * has accumulated enough lookback context:
 *
 * ```
 * round = 0
 * while True:
 *     from_left, from_right = co_await he.exchange(send_left, send_right)
 *     accumulate halos; check if lookback is covered (my_done)
 *     all_done = co_await allreduce(my_done)    // separate AllReduce
 *     if all_done: break
 *     send_right = from_left   // propagate one hop further right
 *     send_left  = from_right  // propagate one hop further left
 * ```
 *
 * ### Tag encoding (uses 4 of the 8 available stage bits per op_id)
 *
 * | Stage | Direction  | Content  |
 * |-------|------------|----------|
 * | 0     | rightward  | metadata |
 * | 1     | rightward  | GPU data |
 * | 2     | leftward   | metadata |
 * | 3     | leftward   | GPU data |
 *
 * Successive rounds reuse the same tags; message ordering is guaranteed by the
 * communicator's no-overtaking property per (rank, tag) pair.
 *
 * @param ctx    Streaming context.
 * @param comm   Communicator.
 * @param op_id  Pre-allocated operation ID. Uses stages 0–3.
 */
class HaloExchange {
  public:
    /**
     * @brief Construct a HaloExchange and register it with the progress thread.
     *
     * @param ctx    Streaming context.
     * @param comm   Communicator.
     * @param op_id  Pre-allocated operation ID. Uses stages 0–3.
     */
    HaloExchange(
        std::shared_ptr<Context> ctx, std::shared_ptr<Communicator> comm, OpID op_id
    );

    ~HaloExchange() noexcept;

    HaloExchange(HaloExchange const&) = delete;
    HaloExchange& operator=(HaloExchange const&) = delete;
    HaloExchange(HaloExchange&&) = delete;
    HaloExchange& operator=(HaloExchange&&) = delete;

    /**
     * @brief Perform one round of bidirectional neighbor exchange.
     *
     * @param send_left  Data to send to rank-1. Pass `std::nullopt` when rank has
     *        no left neighbor (rank == 0) or has nothing to send leftward.
     * @param send_right Data to send to rank+1. Pass `std::nullopt` when rank has
     *        no right neighbor (rank == nranks-1) or has nothing to send rightward.
     *
     * @return Coroutine yielding `(from_left, from_right)`:
     *   - `from_left`:  data received from rank-1; `nullopt` if rank == 0 or
     *                   the left neighbor had nothing to send.
     *   - `from_right`: data received from rank+1; `nullopt` if rank == nranks-1
     *                   or the right neighbor had nothing to send.
     *
     * @note See class-level `@warning` for sequential-use and lifetime requirements.
     */
    coro::task<std::pair<
        std::optional<PackedData>,  // from_left  (from rank-1)
        std::optional<PackedData>  // from_right (from rank+1)
        >>
    exchange(std::optional<PackedData> send_left, std::optional<PackedData> send_right);

  private:
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<Communicator> comm_;
    OpID op_id_;

    mutable std::mutex mutex_;
    coro::event event_{true};  ///< initially set (no active round)

    // Active-round receive state (protected by mutex_)
    bool left_done_{true};
    bool right_done_{true};
    bool left_meta_received_{false};
    bool right_meta_received_{false};
    std::optional<PackedData> from_left_;
    std::optional<PackedData> from_right_;
    std::unique_ptr<Communicator::Future> left_data_recv_future_;
    std::unique_ptr<Communicator::Future> right_data_recv_future_;
    std::unique_ptr<std::vector<std::uint8_t>> left_metadata_;
    std::unique_ptr<std::vector<std::uint8_t>> right_metadata_;
    std::uint64_t left_data_size_{0};
    std::uint64_t right_data_size_{0};

    // Fire-and-forget send futures, cleaned up in event_loop
    std::vector<std::unique_ptr<Communicator::Future>> sends_;

    // Set false by destructor to signal event_loop to return Done
    std::atomic<bool> active_{true};
    ProgressThread::FunctionID function_id_;

    ProgressThread::ProgressState event_loop();
};

}  // namespace rapidsmpf::streaming
