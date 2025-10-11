/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/bcast_node.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>

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
    Message const& msg, std::vector<std::shared_ptr<Channel>>& chs_out
) {
    std::vector<coro::task<bool>> tasks;
    tasks.reserve(chs_out.size());
    for (auto& ch_out : chs_out) {
        tasks.push_back(ch_out->send(msg.shallow_copy()));
    }
    coro_results(co_await coro::when_all(std::move(tasks)));
}
}  // namespace

Node bcast_node(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out,
    BCastPolicy policy
) {
    ShutdownAtExit c1{ch_in};
    ShutdownAtExit c2{chs_out};
    co_await ctx->executor()->schedule();

    switch (policy) {
    case BCastPolicy::BOUNDED:
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            co_await send_to_channels(msg, chs_out);
        }
        break;
    case BCastPolicy::UNBOUNDED:
        // TODO: Instead of buffering all messages before broadcasting,
        //       stream them directly by giving each output channel its own
        //       `coro::queue` and spawning a coroutine per channel that
        //       sends from that queue.
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
                co_await send_to_channels(std::move(msg), chs_out);
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
