/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <coro/task.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
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
 * Multiple calls to `exchange()` on the same instance are safe and represent
 * successive rounds of halo propagation (for multi-hop window coverage when the
 * rolling window period exceeds a single rank's data range).
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
 *     from_left, from_right = co_await he.exchange(send_right, send_left)
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
 * ### Implementation note (TODO for C++ devs)
 *
 * The Python prototype (`halo_exchange.py` in `rapidsmpf/streaming/coll/`) uses two
 * `streaming::AllGather` collectives as a stand-in, which communicates O(N) messages.
 * This C++ class should replace that with direct `Communicator::send/recv`:
 *
 * 1. For each non-null halo: `comm_->send(metadata, neighbor, Tag(op_id_, stage))`
 *    then `comm_->send(data_buffer, neighbor, Tag(op_id_, stage+1))`.
 * 2. For receives: post `recv_sync_host_data` to get the metadata (and learn the
 *    GPU data size), then allocate a device buffer and post `recv` for the data.
 * 3. Drive completion with `coro::event` + progress-thread callbacks, mirroring the
 *    pattern in `streaming::AllGather`.
 *
 * @param ctx    Streaming context.
 * @param comm   Communicator.
 * @param op_id  Pre-allocated operation ID. Uses stages 0–3.
 */
class HaloExchange {
  public:
    HaloExchange(
        std::shared_ptr<Context> ctx,
        std::shared_ptr<Communicator> comm,
        OpID op_id
    );

    ~HaloExchange() noexcept;

    HaloExchange(HaloExchange const&)            = delete;
    HaloExchange& operator=(HaloExchange const&) = delete;
    HaloExchange(HaloExchange&&)                 = delete;
    HaloExchange& operator=(HaloExchange&&)      = delete;

    /**
     * @brief Perform one round of bidirectional neighbor exchange.
     *
     * @param send_right Data to send to rank+1. Pass `std::nullopt` when rank has
     *        no right neighbor (rank == nranks-1) or has nothing to send rightward.
     * @param send_left  Data to send to rank-1. Pass `std::nullopt` when rank has
     *        no left neighbor (rank == 0) or has nothing to send leftward.
     *
     * @return Coroutine yielding `(from_left, from_right)`:
     *   - `from_left`:  data received from rank-1; `nullopt` if rank == 0 or
     *                   the left neighbor had nothing to send.
     *   - `from_right`: data received from rank+1; `nullopt` if rank == nranks-1
     *                   or the right neighbor had nothing to send.
     *
     * @note Successive calls are safe because messages are delivered FIFO per
     *       (sender_rank, tag) pair by the communicator.
     */
    coro::task<std::pair<
        std::optional<PackedData>,   // from_left  (from rank-1)
        std::optional<PackedData>    // from_right (from rank+1)
    >>
    exchange(
        std::optional<PackedData> send_right,
        std::optional<PackedData> send_left
    );

  private:
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<Communicator> comm_;
    OpID op_id_;
};

}  // namespace rapidsmpf::streaming
