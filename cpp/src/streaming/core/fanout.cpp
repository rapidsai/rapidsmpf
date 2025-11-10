/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <deque>
#include <ranges>
#include <stop_token>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Asynchronously send a message to multiple output channels.
 *
 * @param msg The message to broadcast. Each channel receives a shallow
 * copy of the original message.
 * @param chs_out The set of output channels to which the message is sent.
 */
Node send_to_channels(
    Context* ctx, Message const& msg, std::vector<std::shared_ptr<Channel>>& chs_out
) {
    std::vector<coro::task<bool>> tasks;
    tasks.reserve(chs_out.size());

    auto try_memory_types =
        LowerMemoryTypesInclusive(msg.content_description().lowest_memory_type_set());
    for (auto& ch_out : chs_out) {
        // do a reservation for each copy, so that it will fallback to host memory if
        // needed
        auto res = ctx->br()->reserve_or_fail(msg.copy_cost(), try_memory_types);
        tasks.push_back(ch_out->send(msg.copy(res)));
    }
    coro_results(co_await coro::when_all(std::move(tasks)));
}

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * @note Bounded fanout requires all the output channels to consume messages before the
 * next message is sent/ consumed from the input channel.
 *
 * @param ctx The context to use.
 * @param ch_in The input channel to receive messages from.
 * @param chs_out The output channels to send messages to.
 * @return A node representing the bounded fanout operation.
 */
Node bounded_fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out
) {
    ShutdownAtExit c1{ch_in};
    ShutdownAtExit c2{chs_out};
    auto& logger = ctx->logger();
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }

        co_await send_to_channels(ctx.get(), msg, chs_out);
        logger.debug("Sent message ", msg.sequence_number());
    }

    std::vector<coro::task<void>> drain_tasks;
    drain_tasks.reserve(chs_out.size());
    std::ranges::for_each(chs_out, [&](auto& ch) {
        drain_tasks.emplace_back(ch->drain(ctx->executor()));
    });
    coro_results(co_await coro::when_all(std::move(drain_tasks)));
    logger.debug("Completed bounded fanout");
}

/**
 * @brief State for the unbounded fanout.
 */
struct UnboundedFanoutState {
    UnboundedFanoutState(size_t num_channels) : ch_next_idx(num_channels, 0) {}

    coro::mutex mtx;
    // notify send tasks to copy & send messages
    coro::condition_variable data_ready;
    // notify this task to receive more data from the input channel
    coro::condition_variable request_data;
    // set to true when the input channel is fully consumed
    bool input_done{false};
    // messages received from the input channel. We use a deque to avoid references being
    // invalidated by reallocations.
    std::deque<Message> recv_messages;
    // next index to send for each channel
    std::vector<size_t> ch_next_idx;
    // index of the first message to purge
    size_t purge_idx{0};
};

/**
 * @brief Send messages to multiple output channels.
 *
 * @param ctx The context to use.
 * @param idx The index of the task
 * @param ch_out The output channel to send messages to.
 * @param state The state of the unbounded fanout.
 * @return A coroutine representing the task.
 */
Node unbounded_fo_send_task(
    Context& ctx,
    size_t idx,
    std::shared_ptr<Channel>& ch_out,
    UnboundedFanoutState& state
) {
    ShutdownAtExit ch_shutdown{ch_out};
    co_await ctx.executor()->schedule();

    auto& logger = ctx.logger();

    size_t curr_recv_msg_sz = 0;  // current size of the recv_messages deque
    while (true) {
        {
            auto lock = co_await state.mtx.scoped_lock();
            co_await state.data_ready.wait(lock, [&] {
                // irrespective of input_done, update the end_idx to the total number of
                // messages
                curr_recv_msg_sz = state.recv_messages.size();
                return state.input_done || state.ch_next_idx[idx] < curr_recv_msg_sz;
            });
            if (state.input_done && state.ch_next_idx[idx] == curr_recv_msg_sz) {
                // no more messages will be received, and all messages have been sent
                break;
            }
        }

        // now we can copy & send messages in indices [next_idx, end_idx)
        for (size_t i = state.ch_next_idx[idx]; i < curr_recv_msg_sz; i++) {
            auto const& msg = state.recv_messages[i];
            RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");

            // make reservations for each message so that it will fallback to host memory
            // if needed

            auto try_memory_types = LowerMemoryTypesInclusive(
                msg.content_description().lowest_memory_type_set()
            );

            auto res = ctx.br()->reserve_or_fail(msg.copy_cost(), try_memory_types);
            RAPIDSMPF_EXPECTS(
                co_await ch_out->send(msg.copy(res)), "failed to send message"
            );
        }
        logger.trace(
            "sent ", idx, " [", state.ch_next_idx[idx], ", ", curr_recv_msg_sz, ")"
        );

        // now next_idx can be updated to end_idx, and if !input_done, we need to request
        // parent task for more data
        auto lock = co_await state.mtx.scoped_lock();
        state.ch_next_idx[idx] = curr_recv_msg_sz;
        if (state.ch_next_idx[idx] == state.recv_messages.size()) {
            if (state.input_done) {
                break;  // no more messages will be received, and all messages have been
                        // sent
            } else {
                // request more data from the input channel
                lock.unlock();
                co_await state.request_data.notify_one();
            }
        }
    }

    co_await ch_out->drain(ctx.executor());
    logger.debug("Send task ", idx, " completed");
}

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * This is a general purpose implementation which can support consuming messages by any
 * channel. A consumer node can decide to consume all messages from a single channel
 * before moving to the next channel, or it can consume messages from all channels before
 * moving to the next message. When a message has been sent to all output channels, it is
 * purged from the internal deque.
 *
 * @param ctx The context to use.
 * @param ch_in The input channel to receive messages from.
 * @param chs_out The output channels to send messages to.
 * @return A node representing the unbounded fanout operation.
 */
Node unbounded_fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out
) {
    ShutdownAtExit ch_in_shutdown{ch_in};
    ShutdownAtExit chs_out_shutdown{chs_out};
    co_await ctx->executor()->schedule();

    auto& logger = ctx->logger();

    logger.debug("Scheduled unbounded fanout");
    UnboundedFanoutState state(chs_out.size());

    // start send tasks for each output channel
    coro::task_container<coro::thread_pool> tasks(ctx->executor());
    for (size_t i = 0; i < chs_out.size(); i++) {
        RAPIDSMPF_EXPECTS(
            tasks.start(unbounded_fo_send_task(*ctx, i, chs_out[i], state)),
            "failed to start send task"
        );
    }

    // input_done is only set by this task, so reading without lock is safe here
    while (!state.input_done) {
        {
            auto lock = co_await state.mtx.scoped_lock();
            co_await state.request_data.wait(lock, [&] {
                return std::ranges::any_of(state.ch_next_idx, [&](size_t next_idx) {
                    return state.recv_messages.size() == next_idx;
                });
            });
        }

        // receive a message from the input channel
        auto msg = co_await ch_in->receive();

        {  // relock mtx to update input_done/ recv_messages
            auto lock = co_await state.mtx.scoped_lock();
            if (msg.empty()) {
                state.input_done = true;
            } else {
                logger.trace("Received input", msg.sequence_number());
                state.recv_messages.emplace_back(std::move(msg));
            }
        }

        // notify send_tasks to copy & send messages
        co_await state.data_ready.notify_all();

        // purge completed send_tasks. This will reset the messages to empty, so that they
        // release the memory, however the deque is not resized. This guarantees that the
        // indices are not invalidated.
        // intentionally not locking the mtx here, because we only need to know a
        // lower-bound on the last completed idx (ch_next_idx values are monotonically
        // increasing)
        size_t last_completed_idx = std::ranges::min(state.ch_next_idx);
        while (state.purge_idx + 1 < last_completed_idx) {
            state.recv_messages[state.purge_idx].reset();
            state.purge_idx++;
        }
        logger.trace(
            "recv_messages active size: ", state.recv_messages.size() - state.purge_idx
        );
    }

    // Note: there will be some messages to be purged after the loop exits, but we don't
    // need to do anything about them here
    co_await tasks.yield_until_empty();
    logger.debug("Unbounded fanout completed");
}

}  // namespace

Node fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out,
    FanoutPolicy policy
) {
    switch (policy) {
    case FanoutPolicy::BOUNDED:
        return bounded_fanout(std::move(ctx), std::move(ch_in), std::move(chs_out));
    case FanoutPolicy::UNBOUNDED:
        return unbounded_fanout(std::move(ctx), std::move(ch_in), std::move(chs_out));
    default:
        RAPIDSMPF_FAIL("Unknown broadcast policy", std::invalid_argument);
    }
}

}  // namespace rapidsmpf::streaming::node
