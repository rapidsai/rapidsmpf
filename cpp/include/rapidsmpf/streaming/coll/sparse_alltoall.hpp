/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include <coro/event.hpp>
#include <coro/task.hpp>

#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Asynchronous (coroutine) interface to `coll::SparseAlltoall`.
 *
 * Many tasks may insert data concurrently. If multiple tasks insert data, the caller is
 * responsible for arranging that `insert_finished()` is only called after all `insert()`
 * operations have completed. Once `wait()` is awaited, extraction is non-blocking.
 *
 * @warning `insert_finished()` must be called and the collective must have completed,
 * as signalled by `wait()`, before the object is destroyed. Otherwise destruction
 * terminates the process.
 */
class SparseAlltoall {
  public:
    /**
     * @brief Construct an asynchronous sparse all-to-all.
     *
     * @param ctx Streaming context.
     * @param comm Communicator for the collective operation.
     * @param op_id Unique identifier for the collective.
     * @param srcs Ranks this rank expects to receive from.
     * @param dsts Ranks this rank may send to.
     * @param reservation_context Opaque telemetry attribution token for internal
     * reservations.
     */
    SparseAlltoall(
        std::shared_ptr<Context> ctx,
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        std::vector<Rank> srcs,
        std::vector<Rank> dsts,
        ReservationContext reservation_context = 0
    );

    SparseAlltoall(SparseAlltoall const&) = delete;
    SparseAlltoall& operator=(SparseAlltoall const&) = delete;
    SparseAlltoall(SparseAlltoall&&) = delete;
    SparseAlltoall& operator=(SparseAlltoall&&) = delete;

    ~SparseAlltoall() noexcept;

    /**
     * @brief Gets the streaming context associated with this object.
     *
     * @return Shared pointer to context.
     */
    [[nodiscard]] std::shared_ptr<Context> const& ctx() const noexcept;

    /**
     * @brief Gets the communicator associated with this SparseAlltoall.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept;

    /// @copydoc rapidsmpf::coll::SparseAlltoall::insert
    void insert(Rank dst, PackedData&& packed_data);

    /// @copydoc rapidsmpf::coll::SparseAlltoall::insert_finished
    void insert_finished();

    /**
     * @brief Wait for the collective to complete on this rank.
     *
     * The collective completes once every rank has called `insert_finished()` and all
     * data addressed to this rank has arrived. After that, data can be extracted and
     * the `op_id` may be reused.
     *
     * Several coroutines may await `wait()` concurrently.
     *
     * @return Coroutine that completes when the collective has completed.
     */
    [[nodiscard]] coro::task<void> wait();

    /// @copydoc rapidsmpf::coll::SparseAlltoall::extract
    [[nodiscard]] std::vector<PackedData> extract(Rank src);

  private:
    coro::event
        event_{};  ///< Event tracking whether all data has arrived and can be extracted.
    std::shared_ptr<Context> ctx_;  ///< Streaming context.
    coll::SparseAlltoall exchange_;  ///< Underlying sparse all-to-all.
};

}  // namespace rapidsmpf::streaming
