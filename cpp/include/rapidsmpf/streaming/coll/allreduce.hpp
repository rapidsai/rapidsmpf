/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <coro/event.hpp>
#include <coro/task.hpp>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Asynchronous (coroutine) interface to `coll::AllReduce`.
 *
 * A single extraction task must await `extract()` to obtain the result and ensure that
 * the reduction completes.
 */
class AllReduce {
  public:
    /**
     * @brief Construct an asynchronous allreduce.
     *
     * @param ctx Streaming context
     * @param comm The communicator for communication.
     * @param input Local data to contribute to the reduction.
     * @param output Allocated buffer in which to place reduction result. Must be the same
     * size and memory type as `input`. Overwritten with the reduction result (values
     * already in the buffer are ignored).
     * @param op_id Unique operation identifier for this allreduce.
     * @param reduce_operator Type-erased reduction operator to use. See `ReduceOperator`.
     */
    AllReduce(
        std::shared_ptr<Context> ctx,
        std::shared_ptr<Communicator> comm,
        std::unique_ptr<Buffer> input,
        std::unique_ptr<Buffer> output,
        OpID op_id,
        coll::ReduceOperator reduce_operator
    );

    AllReduce(AllReduce const&) = delete;
    AllReduce& operator=(AllReduce const&) = delete;
    AllReduce(AllReduce&&) = delete;
    AllReduce& operator=(AllReduce&&) = delete;

    ~AllReduce() noexcept;

    /**
     * @brief Gets the streaming context associated with this AllReduce object.
     *
     * @return Shared pointer to context.
     */
    [[nodiscard]] std::shared_ptr<Context> const& ctx() const noexcept;

    /**
     * @brief Gets the communicator associated with this AllReduce.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept;

    /**
     * @brief Wait for completion and extract the reduced data.
     *
     * @return Coroutine that completes when the result is available and returns a pair of
     * the two `Buffer`s passed to the constructor. The first `Buffer` contains an
     * implementation-defined value, the second `Buffer` contains the final reduced
     * result.
     */
    coro::task<std::pair<std::unique_ptr<Buffer>, std::unique_ptr<Buffer>>> extract();

  private:
    coro::event
        event_{};  ///< Event tracking whether all data has arrived and can be extracted.
    std::shared_ptr<Context> ctx_;  ///< Streaming context.
    coll::AllReduce reducer_;  ///< Underlying collective allreduce.
};

}  // namespace rapidsmpf::streaming
