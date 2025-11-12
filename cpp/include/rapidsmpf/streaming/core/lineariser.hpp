/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Linearise insertion into an output channel from a fixed number of producers
 * by sequence number.
 *
 * Producers are polled in round-robin fashion, and therefore must deliver messages in
 * round-robin increasing sequence number order. If this guarantee is upheld, then the
 * output of the `Lineariser` is guaranteed to be in total order of the sequence
 * numbers.
 *
 *
 * Example usage:
 * @code{.cpp}
 * auto ctx = std::make_shared<Context>(...);
 * auto ch_out = ctx->create_channel();
 * auto linearise = std::make_shared<Lineariser>(ch_out, 8);
 * std::vector<Node> tasks;
 * // Draining the lineariser will pull from all the input channels until they are
 * // shutdown and send to the output channel until it is consumed.
 * tasks.push_back(linearise->drain());
 * for (auto& ch_in: lineariser->get_inputs()) {
 *   // Each producer promises to send an increasing stream of sequence ids in
 *   // round-robin fashion. That is, if there are P producers, producer 0 sends
 *   // [0, P, 2P, ...], producer 1 sends [1, P+1, 2P + 1, ...] and producer i
 *   // sends [i, P + i, 2P + i, ...].
 *   tasks.push_back(producer(ctx, ch_in, ...));
 * }
 * coro_results(co_await coro::when_all(std::move(tasks)));
 * // ch_out will see inputs in global total order of sequence id.
 * @endcode
 */
class Lineariser {
  public:
    /**
     * @brief Create a new `Lineariser` into an output channel.
     *
     * @param ctx Streaming context.
     * @param ch_out The output channel.
     * @param num_producers The number of producers.
     * @param buffer_size The number of messages that are buffered in the lineariser
     * from each producer.
     */
    Lineariser(
        std::shared_ptr<Context> ctx,
        std::shared_ptr<Channel> ch_out,
        std::size_t num_producers,
        std::size_t buffer_size = 1
    )
        : ctx_{std::move(ctx)}, ch_out_{std::move(ch_out)} {
        queues_.reserve(num_producers);
        for (std::size_t i = 0; i < num_producers; i++) {
            queues_.push_back(ctx->create_bounded_queue(buffer_size));
        }
    }

    /**
     * @brief Get a reference to the input queues.
     *
     * @return Reference to the `BoundedQueue`s to send into.
     *
     * @note Behaviour is undefined if more than one producer coroutine sends into the
     * same queue.
     */
    std::vector<std::shared_ptr<BoundedQueue>>& get_queues() {
        return queues_;
    }

    /**
     * @brief Process inputs and send to the output channel.
     *
     * @return Coroutine representing the linearised sends of all producers.
     *
     * @note This coroutine should be awaited in a `coro::when_all` with all of the
     * producer tasks.
     */
    Node drain() {
        ShutdownAtExit c{ch_out_};
        co_await ctx_->executor()->schedule();
        while (!queues_.empty()) {
            for (auto& q : queues_) {
                auto [receipt, msg] = co_await q->receive();
                if (msg.empty()) {
                    q = nullptr;
                    continue;
                }
                if (!co_await ch_out_->send(std::move(msg))) {
                    // Output channel is shut down, tell the producers to shutdown.
                    break;
                }
                co_await receipt;
            }
            std::erase(queues_, nullptr);
        }
        // We either exited the loop because all the queues are gone, or the output
        // channel is shutdown (in which case we want no more inputs), so either way, just
        // shut down the remaining queues.
        std::vector<coro::task<void>> tasks;
        tasks.reserve(1 + queues_.size());
        for (auto& q : queues_) {
            tasks.push_back(q->shutdown());
        }
        tasks.push_back(ch_out_->drain(ctx_->executor()));
        coro_results(co_await coro::when_all(std::move(tasks)));
    }

    /**
     * @brief Shut down the lineariser, informing both producers and consumer/
     *
     * @return Coroutine representing the shutdown of all input queues and the output
     * channel.
     */
    coro::task<void> shutdown() {
        std::vector<coro::task<void>> tasks;
        tasks.reserve(1 + queues_.size());
        for (auto& q : queues_) {
            tasks.push_back(q->shutdown());
        }
        tasks.push_back(ch_out_->shutdown());
        coro_results(co_await coro::when_all(std::move(tasks)));
    }

  private:
    std::shared_ptr<Context> ctx_;
    std::vector<std::shared_ptr<BoundedQueue>> queues_;
    std::shared_ptr<Channel> ch_out_;  ///< Output channel.
};

}  // namespace rapidsmpf::streaming
