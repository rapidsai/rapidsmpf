/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <deque>
#include <stop_token>

#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>

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
    for (auto& ch_out : chs_out) {
        // do a reservation for each copy, so that it will fallback to host memory if
        // needed
        auto res = ctx->br()->reserve_or_fail(msg.copy_cost());
        tasks.push_back(ch_out->send(msg.copy(res)));
    }
    coro_results(co_await coro::when_all(std::move(tasks)));
}

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

    for (auto& ch : chs_out) {
        co_await ch->drain(ctx->executor());
    }
    logger.debug("Completed bounded fanout");
}

Node unbounded_fo_send_task(
    Context& ctx,
    size_t idx,
    std::vector<std::shared_ptr<Channel>> const& chs_out,
    std::vector<size_t>& ch_next_idx,
    coro::mutex& mtx,
    coro::condition_variable& data_ready,
    coro::condition_variable& request_data,
    bool const& input_done,
    std::deque<Message> const& recv_messages
) {
    auto& logger = ctx.logger();
    ShutdownAtExit ch_shutdown{chs_out[idx]};
    co_await ctx.executor()->schedule();

    size_t curr_recv_msg_sz;
    while (true) {
        {
            auto lock = co_await mtx.scoped_lock();
            co_await data_ready.wait(lock, [&] {
                // irrespective of input_done, update the end_idx to the total number of
                // messages
                curr_recv_msg_sz = recv_messages.size();
                return input_done || ch_next_idx[idx] < curr_recv_msg_sz;
            });
            if (input_done && ch_next_idx[idx] == curr_recv_msg_sz) {
                // no more messages will be received, and all messages have been sent
                break;
            }
        }

        // now we can copy & send messages in indices [next_idx, end_idx)
        for (size_t i = ch_next_idx[idx]; i < curr_recv_msg_sz; i++) {
            auto const& msg = recv_messages[i];
            RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");

            // make reservations for each message so that it will fallback to host memory
            // if needed
            auto res = ctx.br()->reserve_or_fail(msg.copy_cost());
            RAPIDSMPF_EXPECTS(
                co_await chs_out[idx]->send(msg.copy(res)), "failed to send message"
            );
        }
        logger.debug("sent ", idx, " [", ch_next_idx[idx], ", ", curr_recv_msg_sz, ")");

        // now next_idx can be updated to end_idx, and if !input_done, we need to request
        // parent task for more data
        auto lock = co_await mtx.scoped_lock();
        ch_next_idx[idx] = curr_recv_msg_sz;
        if (input_done && ch_next_idx[idx] == recv_messages.size()) {
            break;
        } else {
            lock.unlock();
            co_await request_data.notify_one();
        }
    }

    co_await chs_out[idx]->drain(ctx.executor());
    logger.debug("Send task ", idx, " completed");
}

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
    coro::mutex mtx;
    coro::condition_variable data_ready;  // notify send tasks to copy & send messages
    coro::condition_variable
        request_data;  // notify this task to receive more data from the input channel
    bool input_done{false};  // set to true when the input channel is fully consumed
    std::deque<Message> recv_messages;  // messages received from the input channel. We
                                        // use a deque to avoid references being
                                        // invalidated by reallocations.
    std::vector<size_t> ch_next_idx(
        chs_out.size(), 0
    );  // next index to send for each channel

    coro::task_container<coro::thread_pool> tasks(ctx->executor());
    for (size_t i = 0; i < chs_out.size(); i++) {
        RAPIDSMPF_EXPECTS(
            tasks.start(unbounded_fo_send_task(
                *ctx,
                i,
                chs_out,
                ch_next_idx,
                mtx,
                data_ready,
                request_data,
                input_done,
                recv_messages
            )),
            "failed to start send task"
        );
    }

    size_t purge_idx = 0;  // index of the first message to purge

    // input_done is only set by this task, so reading without lock is safe here
    while (!input_done) {
        {
            auto lock = co_await mtx.scoped_lock();
            co_await request_data.wait(lock, [&] {
                // return recv_messages.size() <= std::ranges::max(ch_next_idx);
                return std::ranges::any_of(ch_next_idx, [&](size_t next_idx) {
                    return recv_messages.size() == next_idx;
                });
            });
        }

        // receive a message from the input channel
        auto msg = co_await ch_in->receive();

        {  // relock mtx to update input_done/ recv_messages
            auto lock = co_await mtx.scoped_lock();
            if (msg.empty()) {
                input_done = true;
            } else {
                logger.debug("Received input", msg.sequence_number());
                recv_messages.emplace_back(std::move(msg));
            }
        }
        
        // notify send_tasks to copy & send messages
        co_await data_ready.notify_all();

        // purge completed send_tasks
        // intentionally not locking the mtx here, because we only need to know a
        // lower-bound on the last completed idx (ch_next_idx values are monotonically
        // increasing)
        size_t last_completed_idx = std::ranges::min(ch_next_idx);
        while (purge_idx + 1 < last_completed_idx) {
            recv_messages[purge_idx].reset();
            purge_idx++;
        }
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
