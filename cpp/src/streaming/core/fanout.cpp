/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

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
        tasks.push_back(ch_out->send(msg.copy(ctx->br(), res)));
    }
    coro_results(co_await coro::when_all(std::move(tasks)));
}

struct UnboundedFanoutState {
    UnboundedFanoutState(std::vector<std::shared_ptr<Channel>>&& chs_out_)
        : chs_out{std::move(chs_out_)},
          ch_next_idx{chs_out.size(), 0},
          ch_data_avail{chs_out.size(), {}},
          send_tasks{chs_out.size()} {}

    coro::task<void> receive_done() {
        auto lock = co_await mtx.scoped_lock();
        n_msgs = chs_out.size();
    }

    [[nodiscard]] constexpr bool all_received() const {
        return n_msgs != std::numeric_limits<size_t>::max();
    }

    [[nodiscard]] constexpr bool all_sent(size_t i) const {
        return all_received() && ch_next_idx[i] == n_msgs;
    }

    // [[nodiscard]] constexpr size_t last_completed_idx() const {
    //     return std::ranges::min(ch_next_idx);
    // }

    // thread-safe data for each send task
    std::vector<std::shared_ptr<Channel>> chs_out;
    std::vector<size_t> ch_next_idx;  // values are strictly increasing
    std::vector<coro::event> ch_data_avail;

    std::vector<Node> send_tasks;

    std::vector<Message> recv_messages;

    coro::mutex mtx;
    coro::condition_variable cv;
    size_t n_msgs{std::numeric_limits<size_t>::max()};
};

Node send_task(Context* ctx, UnboundedFanoutState& state, size_t i) {
    co_await ctx->executor()->schedule();

    // co_await state.data_avail;  // wait for data to be available

    while (true) {
        // wait for the data to be available
        co_await state.ch_data_avail[i];

        if (state.all_sent(i)) {
            // all messages have been sent, nothing else to do
            break;
        }

        auto const& msg = state.recv_messages[state.ch_next_idx[i]];
        // copy msg
        // msg.content_size

        {
            auto lock = state.mtx.scoped_lock();
            state.ch_next_idx[i]++;
        }
        co_await state.cv.notify_one();


        //
        if (state.ch_next_idx[i] == state.recv_messages.size() && !state.all_received()) {
            state.ch_data_avail[i].reset();
        }
    }
}

Node unbounded_fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out
) {
    ShutdownAtExit c{ch_in};
    ShutdownAtExit c2{chs_out};
    co_await ctx->executor()->schedule();

    UnboundedFanoutState state(std::move(chs_out));

    size_t purge_idx = 0;
    while (true) {
        {
            auto lock = co_await state.mtx.scoped_lock();
            co_await state.cv.wait(lock, [&] {
                return state.recv_messages.size() <= std::ranges::max(state.ch_next_idx);
            });
        }

        // n_msgs is only set by this task. So, reading w/o a lock is safe.
        if (state.n_msgs == std::numeric_limits<size_t>::max()) {
            auto msg = co_await ch_in->receive();
            auto lock = co_await state.mtx.scoped_lock();
            if (msg.empty()) {
                // no more messages to receive
                state.n_msgs = state.recv_messages.size();
            } else {
                state.recv_messages.push_back(std::move(msg));
                lock.unlock();

                for (auto& event : state.ch_data_avail) {
                    event.set();
                }
            }
        }

        size_t last_completed_idx = std::ranges::min(state.ch_next_idx);
        while (purge_idx <= last_completed_idx) {
            state.ch_data_avail[purge_idx].reset();
            purge_idx++;
        }
    }
}


}  // namespace

Node fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out,
    FanoutPolicy policy
) {
    ShutdownAtExit c1{ch_in};
    ShutdownAtExit c2{chs_out};
    co_await ctx->executor()->schedule();

    switch (policy) {
    case FanoutPolicy::BOUNDED:
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            co_await send_to_channels(ctx.get(), msg, chs_out);
        }
        break;
    case FanoutPolicy::UNBOUNDED:
        {
            // First we receive until the input channel is shutdown.
            std::vector<Message> messages;
            while (true) {
                auto msg = co_await ch_in->receive();
                if (msg.empty()) {
                    break;
                }
                messages.push_back(std::move(msg));
            }
            // Then we send each input message to all output channels.
            for (auto& msg : messages) {
                co_await send_to_channels(ctx.get(), msg, chs_out);
            }
            break;
        }
    default:
        RAPIDSMPF_FAIL("Unknown broadcast policy", std::invalid_argument);
    }

    // Finally, we drain all output channels.
    std::vector<Node> tasks;
    tasks.reserve(chs_out.size());
    for (auto& ch_out : chs_out) {
        tasks.push_back(ch_out->drain(ctx->executor()));
    }
    coro_results(co_await coro::when_all(std::move(tasks)));
}

}  // namespace rapidsmpf::streaming::node