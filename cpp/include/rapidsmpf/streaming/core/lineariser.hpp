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
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Linearise insertion into an output channel from a fixed number of producers by
 * sequence number.
 *
 * Producers are polled in round-robin fashion, and hence, given `P` producers and `N`
 * ordered tasks, producer `i` _must_ deliver the strided range of tasks `tasks[i::P]`.
 *
 * @warning Individual producers promise to send messages into their channel in strictly
 * increasing sequence number order. This bounds the buffering required in the
 * `Lineariser` to the number of distinct producers. If this precondition is not met,
 * behaviour is undefined.
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
 *   // Each producer promises to send an increasing stream of sequence ids.
 *   // For best performance they should attempt to take "interleaved" ids from a shared
 *   // task list. That is, don't have producer-0 produce the first K ids, producer-1 the
 *   // next K, and so forth.
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
     */
    Lineariser(
        std::shared_ptr<Context> ctx,
        std::shared_ptr<Channel> ch_out,
        std::size_t num_producers
    )
        : ctx_{std::move(ctx)}, ch_out_{std::move(ch_out)} {
        inputs_.reserve(num_producers);
        for (std::size_t i = 0; i < num_producers; i++) {
            inputs_.emplace_back(std::make_unique<Semaphore>(1), ctx_->create_channel());
        }
    }

    /**
     * @brief Get a reference to the input channels.
     *
     * @return Reference to the `Channel`s to send into.
     *
     * @note Behaviour is undefined if more than one producer coroutine sends into the
     * same channel.
     */
    std::vector<std::pair<std::unique_ptr<Semaphore>, std::shared_ptr<Channel>>>&
    get_inputs() {
        return inputs_;
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
        while (!inputs_.empty()) {
            for (auto& input : inputs_) {
                auto& [sem, ch_in] = input;
                auto msg = co_await ch_in->receive();
                if (msg.empty()) {
                    input = {nullptr, nullptr};
                    continue;
                }
                co_await ch_out_->send(std::move(msg));
                co_await sem->release();
            }
            std::erase(
                inputs_,
                std::pair<std::unique_ptr<Semaphore>, std::shared_ptr<Channel>>{
                    nullptr, nullptr
                }
            );
        }
        co_await ch_out_->drain(ctx_->executor());
    }

  private:
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<Channel> ch_out_;  ///< Output channel.
    std::vector<std::pair<std::unique_ptr<Semaphore>, std::shared_ptr<Channel>>>
        inputs_;  ///< Input channels.
};

}  // namespace rapidsmpf::streaming
