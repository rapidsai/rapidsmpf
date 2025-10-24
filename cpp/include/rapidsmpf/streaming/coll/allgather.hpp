/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rapidsmpf/allgather/allgather.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/chunks/packed_data.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <coro/event.hpp>
#include <coro/task.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Asynchronous (coroutine) interface to `allgather::AllGather`.
 *
 * Once the AllGather is created, many tasks may insert data into it. If multiple tasks
 * insert data, the user is responsible for arranging that `insert_finished` is only
 * called after all `insert`ions have completed. A single consumer task should extract
 * data.
 */
class AllGather {
  public:
    /// @copydoc allgather::AllGather::Ordered
    using Ordered = rapidsmpf::allgather::AllGather::Ordered;
    /**
     * @brief Construct an asynchronous allgather.
     *
     * @param ctx Streaming context
     * @param op_id Unique identifier for the allgather.
     */
    AllGather(std::shared_ptr<Context> ctx, OpID op_id);

    AllGather(AllGather const&) = delete;
    AllGather& operator=(AllGather const&) = delete;
    AllGather(AllGather&&) = delete;
    AllGather& operator=(AllGather&&) = delete;

    ~AllGather();

    /**
     * @brief Gets the streaming context associated with this AllGather object.
     *
     * @return Shared pointer to context.
     */
    [[nodiscard]] std::shared_ptr<Context> ctx() const noexcept;

    /**
     * @brief Insert a chunk into the allgather.
     *
     * @param sequence_number The sequence number for this chunk.
     * @param chunk The chunk to insert.
     */
    void insert(std::uint64_t sequence_number, PackedDataChunk&& chunk);

    /// @copydoc rapidsmpf::allgather::AllGather::insert_finished()
    void insert_finished();

    /**
     * @brief Extract all gathered data.
     *
     * @param ordered If the extracted data should be ordered. If ordered, return data
     * will be ordered first by rank and then by sequence number of the inserted chunks on
     * that rank.
     *
     * @return Coroutine that completes when all data is available for extraction and
     * returns the data.
     */
    coro::task<std::vector<PackedDataChunk>> extract_all(Ordered ordered = Ordered::YES);

  private:
    coro::event
        event_{};  ///< Event tracking whether all data has arrived and can be extracted.
    std::shared_ptr<Context> ctx_;  ///< Streaming context.
    allgather::AllGather gatherer_;  ///< Underlying collective allgather.
};

namespace node {

/**
 * @brief Create an allgather node for a single allgather operation.
 *
 * This is a streaming version of `rapidsmpf::allgather::AllGather` that operates on
 * packed data received through `Channel`s.
 *
 * @param ctx The streaming context to use.
 * @param ch_in Input channel providing `PackedDataChunk`s to be gathered.
 * @param ch_out Output channel where the gathered `PackedDataChunk`s are sent.
 * @param op_id Unique identifier for the operation.
 * @param ordered If the extracted data should be sent to the output channel with sequence
 * numbers corresponding to the global total order of input chunks. If yes, then the
 * sequence numbers of the extracted data will be ordered first by rank and then by input
 * sequence number. If no, the sequence number of the extracted chunks will have no
 * relation to any input sequence order.
 *
 * @return A streaming node that completes when the allgather is finished and the output
 * channel is drained.
 */
Node allgather(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    AllGather::Ordered ordered = AllGather::Ordered::YES
);
}  // namespace node
}  // namespace rapidsmpf::streaming
