/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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
    std::shared_ptr<Context>&& ctx,
    std::shared_ptr<Channel>&& ch_in,
    std::vector<std::shared_ptr<Channel>>&& chs_out
) {
    ShutdownAtExit c1{ch_in};
    ShutdownAtExit c2{chs_out};
    co_await ctx->executor()->schedule();

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await send_to_channels(ctx.get(), msg, chs_out);
    }

    // Finally, we drain all output channels.
    std::vector<Node> tasks;
    tasks.reserve(chs_out.size());
    std::ranges::transform(chs_out, std::back_inserter(tasks), [&](auto& ch_out) {
        return ch_out->drain(ctx->executor());
    });

    coro_results(co_await coro::when_all(std::move(tasks)));
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
    std::vector<Message> const& recv_messages
) {
    ShutdownAtExit c{chs_out[idx]};
    co_await ctx.executor()->schedule();
    size_t end_idx;
    while (true) {
        {
            auto lock = co_await mtx.scoped_lock();
            co_await data_ready.wait(lock, [&] {
                // irrespective of input_done, update the end_idx to the total number of
                // messages
                end_idx = recv_messages.size();
                return input_done || ch_next_idx[idx] < end_idx;
            });
            if (input_done && ch_next_idx[idx] == end_idx) {
                break;
            }
        }

        // now we can copy & send messages in indices [next_idx, end_idx)
        for (size_t i = ch_next_idx[idx]; i < end_idx; i++) {
            auto const& msg = recv_messages[i];

            // make reservations for each message so that it will fallback to host memory
            // if needed
            auto res = ctx.br()->reserve_or_fail(msg.copy_cost());
            co_await chs_out[idx]->send(msg.copy(res));
        }

        // now next_idx can be updated to end_idx, and if !input_done, we need to request
        // parent task for more data
        auto lock = co_await mtx.scoped_lock();
        ch_next_idx[idx] = end_idx;
        if (input_done && ch_next_idx[idx] == end_idx) {
            break;
        } else {
            lock.unlock();
            co_await request_data.notify_one();
        }
    }

    co_await chs_out[idx]->drain(ctx.executor());
}

Node unbounded_fanout(
    std::shared_ptr<Context>&& ctx,
    std::shared_ptr<Channel>&& ch_in,
    std::vector<std::shared_ptr<Channel>>&& chs_out
) {
    ShutdownAtExit c{ch_in};
    ShutdownAtExit c2{chs_out};
    co_await ctx->executor()->schedule();


    coro::mutex mtx;
    coro::condition_variable data_ready;
    coro::condition_variable request_data;
    bool input_done{false};
    std::vector<Message> recv_messages;

    std::vector<size_t> ch_next_idx(chs_out.size(), 0);

    std::vector<Node> tasks;
    tasks.reserve(chs_out.size());
    for (size_t i = 0; i < chs_out.size(); i++) {
        tasks.emplace_back(unbounded_fo_send_task(
            *ctx,
            i,
            chs_out,
            ch_next_idx,
            mtx,
            data_ready,
            request_data,
            input_done,
            recv_messages
        ));
    }

    size_t purge_idx = 0;
    // input_done is only set by this task, so reading without lock is safe here
    while (!input_done) {
        {
            auto lock = co_await mtx.scoped_lock();
            co_await request_data.wait(lock, [&] {
                return std::ranges::any_of(ch_next_idx, [&](size_t next_idx) {
                    return recv_messages.size() >= next_idx;
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
    coro_results(co_await coro::when_all(std::move(tasks)));
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
        co_await bounded_fanout(std::move(ctx), std::move(ch_in), std::move(chs_out));
        break;
    case FanoutPolicy::UNBOUNDED:
        co_await unbounded_fanout(std::move(ctx), std::move(ch_in), std::move(chs_out));
        break;
    default:
        RAPIDSMPF_FAIL("Unknown broadcast policy", std::invalid_argument);
    }
}

}  // namespace rapidsmpf::streaming::node
