/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <queue>
#include <utility>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Linearise insertion into an output channel from a fixed number of producers by
 * sequence number.
 *
 * Channels are guaranteed to deliver data in increasing sequence number order. When we
 * have multiple producers we must ensure that they queue up their productions into the
 * output channel in sequence number order. The `Lineariser` provides this interface by
 * providing a per-producer "output" channel and buffering appropriately.
 *
 * @warning Individual producers promise to send messages into their channel in strictly
 * increasing sequence number order. This bounds the buffering required in the
 * `Lineariser` to the number of distinct producers. If this precondition is not met,
 * behaviour is undefined.
 */
class Lineariser {
  public:
    /**
     * @brief Create a new `Lineariser` into an output channel.
     *
     * @param ch_out The output channel
     * @param num_producers The number of producers.
     */
    Lineariser(std::shared_ptr<Channel> ch_out, std::size_t num_producers)
        : ch_out_{std::move(ch_out)} {
        inputs_.reserve(num_producers);
        for (std::size_t i = 0; i < num_producers; i++) {
            inputs_.push_back(std::make_shared<Channel>());
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
    std::vector<std::shared_ptr<Channel>>& get_inputs() {
        return inputs_;
    }

    /**
     * @brief Process inputs and send to the output channel.
     *
     * @param ctx The execution context to use.
     * @return Coroutine representing the linearised sends of all producers.
     *
     * @note This coroutine should be awaited in a `coro::when_all` with all of the
     * producer tasks.
     */
    Node drain(std::shared_ptr<Context> ctx) {
        ShutdownAtExit c{ch_out_};
        co_await ctx->executor()->schedule();
        // Invariant: the heap always contains exactly zero-or-one messages from each
        // producer. We always extract the minimum element, and then repoll the producer
        // that made that element. First poll all producers
        for (std::size_t p = 0; p < inputs_.size(); p++) {
            auto msg = co_await inputs_[p]->receive();
            if (!msg.empty()) {
                min_heap_.emplace(p, std::move(msg));
            }
        }
        while (true) {
            co_await ctx->executor()->schedule();
            if (min_heap_.empty()) {
                // All producers have shut down.
                co_return;
            } else {
                auto item = min_heap_.pop_top();
                co_await ch_out_->send(std::move(item.value));
                // And refill from the producer we just consumed from.
                auto msg = co_await inputs_[item.producer]->receive();
                if (!msg.empty()) {
                    min_heap_.emplace(item.producer, std::move(msg));
                }
            }
        }
        co_await ch_out_->drain(ctx->executor());
    }

  private:
    /**
     * @brief Tracking struct for message plus the id of the producer.
     */
    struct Item {
        std::size_t producer;  ///< Which producer this message was from.
        Message value;  ///< The message.
    };

    /**
     * Comparator of items for min heap such that items with the lowest sequence number
     * come first.
     */
    struct Comparator {
        bool operator()(Item& l, Item& r) {
            // std::priority_queue is a max_heap, hence greater
            return l.value.sequence_number() > r.value.sequence_number();
        }
    };

    struct min_heap : std::priority_queue<Item, std::vector<Item>, Comparator> {
      public:
        Item pop_top() {
            std::ranges::pop_heap(c, comp);
            auto value = std::move(c.back());
            c.pop_back();
            return value;
        }

      protected:
        using std::priority_queue<Item, std::vector<Item>, Comparator>::c;
        using std::priority_queue<Item, std::vector<Item>, Comparator>::comp;
    };

    std::shared_ptr<Channel> ch_out_;  ///< Output channel.
    std::vector<std::shared_ptr<Channel>> inputs_;  ///< Input channels.
    min_heap min_heap_{};  ///< Heap of to be sent messages.
};

}  // namespace rapidsmpf::streaming
