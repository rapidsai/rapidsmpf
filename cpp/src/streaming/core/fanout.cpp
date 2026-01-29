/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <deque>
#include <ranges>
#include <span>

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/core/spillable_messages.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Asynchronously send a message to multiple output channels.
 *
 * @param msg The message to broadcast. Each channel receives a deep copy of the original
 * message.
 * @param chs_out The set of output channels to which the message is sent.
 */
Node send_to_channels(
    Context& ctx, Message&& msg, std::vector<std::shared_ptr<Channel>>& chs_out
) {
    RAPIDSMPF_EXPECTS(!chs_out.empty(), "output channels cannot be empty");

    auto async_copy_and_send = [](Context& ctx_,
                                  Message const& msg_,
                                  size_t msg_sz_,
                                  Channel& ch_) -> coro::task<bool> {
        co_await ctx_.executor()->schedule();
        auto const& cd = msg_.content_description();
        auto const mem_types = leq_memory_types(cd.principal_memory_type());
        auto res = ctx_.br()->reserve_or_fail(msg_sz_, mem_types);
        co_return co_await ch_.send(msg_.copy(res));
    };

    // async copy & send tasks for all channels except the last one
    std::vector<coro::task<bool>> async_send_tasks;
    async_send_tasks.reserve(chs_out.size() - 1);
    size_t msg_sz = msg.copy_cost();
    for (size_t i = 0; i < chs_out.size() - 1; i++) {
        async_send_tasks.emplace_back(async_copy_and_send(ctx, msg, msg_sz, *chs_out[i]));
    }

    // note that the send tasks may return false if the channel is shut down. But we can
    // safely ignore this in bounded fanout.
    coro_results(co_await coro::when_all(std::move(async_send_tasks)));

    // move the message to the last channel to avoid extra copy
    co_await chs_out.back()->send(std::move(msg));
}

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * @note Bounded fanout requires all the output channels to consume messages before
 * the next message is sent/consumed from the input channel.
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
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }

        // filter out shut down channels to avoid making unnecessary copies
        std::erase_if(chs_out, [](auto&& ch) { return ch->is_shutdown(); });
        if (chs_out.empty()) {
            // all channels are shut down, so we can break & shutdown the input channel
            break;
        }
        co_await send_to_channels(*ctx, std::move(msg), chs_out);
    }

    std::vector<Node> drain_tasks;
    drain_tasks.reserve(chs_out.size());
    for (auto& ch : chs_out) {
        drain_tasks.emplace_back(ch->drain(ctx->executor()));
    }
    coro_results(co_await coro::when_all(std::move(drain_tasks)));
}

/**
 * @brief Unbounded fanout implementation.
 *
 * The implementation follows a pull-based model, where the send tasks request data from
 * the recv task. There is one recv task that receives messages from the input channel,
 * and there are N send tasks that send messages to the output channels.
 *
 * Main task operation:
 * - There is a shared deque of cached messages, and a vector that indicates the next
 * index of the message to be sent to each output channel.
 * - All shared resources are protected by a mutex. There are two condition variables
 * where:
 *   - recv task notifies send tasks when new messages are cached
 *   - send tasks notify recv task when they have completed sending messages
 * - Recv task awaits until the number of cached messages at least one send task has
 * completed sending all the cached messages. It will then pull a message from the input
 * channel, cache it, and notify the send tasks about the new messages. recv task
 * continues this process until the input channel is fully consumed.
 * - Each send task awaits until there are more cached messages to send. When the new
 * messages available noitification is received, it will continue to copy and send cached
 * messages, starting from the index of the last sent message, to the end of the cached
 * messages (as it last observed). Then it updates the last completed message index and
 * notifies the recv task. This process continues until the recv task notifies that the
 * input channel is fully consumed.
 *
 * Additional considerations:
 * - In the recv task loop, it also identifies the lowest completed message index by all
 * send tasks. Message upto this index are no longer needed, and are released from the
 * cached messages deque.
 * - When a send task fails to send a message, this means the channel may have been
 * prematurely shut down. In this case, it sets a sential value to mark it as invalid.
 * Recv task will filter out channels with the invalid sentinel value.
 * - There are two RAII helpers to ensure that the notification mechanisms are properly
 * cleaned up when the unbounded fanout state goes out of scope/ encounters an error.
 *
 */
struct UnboundedFanout {
    /**
     * @brief Constructor.
     *
     * @param num_channels The number of output channels.
     */
    explicit UnboundedFanout(size_t num_channels) : per_ch_processed(num_channels, 0) {}

    /**
     * @brief Sentinel value indicating that the index is invalid. This is set when a
     * failure occurs during send tasks. process input task will filter out messages with
     * this index.
     */
    static constexpr size_t InvalidIdx = std::numeric_limits<size_t>::max();

    /**
     * @brief RAII helper class to set a channel index to invalid and notify the process
     * input task to check if it should break.
     */
    struct SetChannelIdxInvalidAtExit {
        UnboundedFanout* fanout;
        size_t& self_next_idx;

        ~SetChannelIdxInvalidAtExit() {
            coro::sync_wait(set_channel_idx_invalid());
        }

        Node set_channel_idx_invalid() {
            if (self_next_idx != InvalidIdx) {
                {
                    auto lock = co_await fanout->mtx.scoped_lock();
                    self_next_idx = InvalidIdx;
                }
                co_await fanout->request_data.notify_one();
            }
        }
    };

    /**
     * @brief Send messages to multiple output channels.
     *
     * @param ctx The context to use.
     * @param self_next_idx Next index to send for the current channel (passed by ref
     * because it needs to be updated)
     * @param ch_out The output channel to send messages to.
     * @return A coroutine representing the task.
     */
    Node send_task(Context& ctx, size_t& self_next_idx, std::shared_ptr<Channel> ch_out) {
        ShutdownAtExit ch_shutdown{ch_out};
        SetChannelIdxInvalidAtExit set_ch_idx_invalid{
            .fanout = this, .self_next_idx = self_next_idx
        };
        co_await ctx.executor()->schedule();

        auto spillable_messages = ctx.spillable_messages();

        size_t n_available_messages = 0;
        std::vector<SpillableMessages::MessageId> msg_ids_to_send;
        while (true) {
            {
                auto lock = co_await mtx.scoped_lock();
                co_await data_ready.wait(lock, [&] {
                    // irrespective of no_more_input, update the end_idx to the total
                    // number of messages
                    n_available_messages = recv_msg_ids.size();
                    return no_more_input || self_next_idx < n_available_messages;
                });
                if (no_more_input && self_next_idx == n_available_messages) {
                    // no more messages will be received, and all messages have been sent
                    break;
                }
                // copy msg ids to send under the lock
                msg_ids_to_send.reserve(n_available_messages - self_next_idx);
                std::ranges::copy(
                    std::ranges::drop_view(
                        recv_msg_ids, static_cast<std::ptrdiff_t>(self_next_idx)
                    ),
                    std::back_inserter(msg_ids_to_send)
                );
            }

            for (auto const msg_id : msg_ids_to_send) {
                auto const cd = spillable_messages->get_content_description(msg_id);
                // Reserve memory for the output using the input message's memory type, or
                // a lower-priority type if needed.
                auto const mem_types = leq_memory_types(cd.principal_memory_type());
                auto res = ctx.br()->reserve_or_fail(cd.content_size(), mem_types);
                if (!co_await ch_out->send(spillable_messages->copy(msg_id, res))) {
                    // Failed to send message. Could be that the channel is shut down.
                    // So we need to abort the send task, and notify the process input
                    // task
                    co_await set_ch_idx_invalid.set_channel_idx_invalid();
                    co_return;
                }
            }
            msg_ids_to_send.clear();

            // now next_idx can be updated to end_idx, and if !no_more_input, we need to
            // request the recv task for more data
            auto lock = co_await mtx.scoped_lock();
            self_next_idx = n_available_messages;
            if (self_next_idx == recv_msg_ids.size()) {
                if (no_more_input) {
                    // no more messages will be received, and all messages have been sent
                    break;
                } else {
                    // request more data from the input channel
                    lock.unlock();
                    co_await request_data.notify_one();
                }
            }
        }
        co_await ch_out->drain(ctx.executor());
    }

    /**
     * @brief RAII helper class to set no_more_input to true and notify all send tasks to
     * wind down when the unbounded fanout state goes out of scope.
     */
    struct SetInputDoneAtExit {
        UnboundedFanout* fanout;

        ~SetInputDoneAtExit() {
            coro::sync_wait(set_input_done());
        }

        // forcibly set no_more_input to true and notify all send tasks to wind down
        Node set_input_done() {
            {
                auto lock = co_await fanout->mtx.scoped_lock();
                fanout->no_more_input = true;
            }
            co_await fanout->data_ready.notify_all();
        }
    };

    /**
     * @brief Wait for a data request from the send tasks.
     *
     * @return A minmax pair of `per_ch_processed` values. min is index of the last
     * completed message index + 1 and max is the index of the latest processed message
     * index + 1. If both are InvalidIdx, it means that all send tasks are in an invalid
     * state.
     */
    auto wait_for_data_request() -> coro::task<std::pair<size_t, size_t>> {
        size_t per_ch_processed_min = InvalidIdx;
        size_t per_ch_processed_max = InvalidIdx;

        auto lock = co_await mtx.scoped_lock();
        co_await request_data.wait(lock, [&] {
            auto filtered_view = std::ranges::filter_view(
                per_ch_processed, [](size_t idx) { return idx != InvalidIdx; }
            );

            auto it = std::ranges::begin(filtered_view);  // advance to first valid idx
            auto end = std::ranges::end(filtered_view);
            if (it == end) {
                // no valid indices, so all send tasks are in an invalid state
                return true;
            }

            auto [min_it, max_it] = std::minmax_element(it, end);
            per_ch_processed_min = *min_it;
            per_ch_processed_max = *max_it;

            return per_ch_processed_max == recv_msg_ids.size();
        });

        co_return std::make_pair(per_ch_processed_min, per_ch_processed_max);
    }

    /**
     * @brief Process input messages and notify send tasks to copy & send messages.
     *
     * @param ctx The context to use.
     * @param ch_in The input channel to receive messages from.
     * @return A coroutine representing the task.
     */
    Node recv_task(Context& ctx, std::shared_ptr<Channel> ch_in) {
        ShutdownAtExit ch_in_shutdown{ch_in};
        SetInputDoneAtExit set_input_done{.fanout = this};
        co_await ctx.executor()->schedule();

        // index of the first message to purge
        size_t purge_idx = 0;

        // To make staged input messages spillable, we insert them into the Context's
        // spillable_messages container while they are in transit.
        auto spillable_messages = ctx.spillable_messages();

        // no_more_input is only set by this task, so reading without lock is safe here
        while (!no_more_input) {
            auto [per_ch_processed_min, per_ch_processed_max] =
                co_await wait_for_data_request();
            if (per_ch_processed_min == InvalidIdx && per_ch_processed_max == InvalidIdx)
            {
                break;
            }

            // receive a message from the input channel
            auto msg = co_await ch_in->receive();

            {
                auto lock = co_await mtx.scoped_lock();
                if (msg.empty()) {
                    no_more_input = true;
                } else {
                    recv_msg_ids.emplace_back(spillable_messages->insert(std::move(msg)));
                }
            }

            // notify send_tasks to copy & send messages
            co_await data_ready.notify_all();

            // Reset messages that are no longer needed, so that they release the memory.
            // However the deque is not resized. This guarantees that the indices are not
            // invalidated.
            while (purge_idx < per_ch_processed_min) {
                std::ignore = spillable_messages->extract(recv_msg_ids[purge_idx]);
                purge_idx++;
            }
        }

        co_await ch_in->drain(ctx.executor());
    }

    coro::mutex mtx;

    /// @brief recv task notifies send tasks to copy & send messages
    coro::condition_variable data_ready;

    /// @brief send tasks notify recv task to pull more data from the input channel
    coro::condition_variable request_data;

    /// @brief set to true when the input channel is fully consumed
    bool no_more_input{false};

    /// @brief messages received from the input channel. Using a deque to avoid
    /// invalidating references by reallocations.
    std::deque<SpillableMessages::MessageId> recv_msg_ids;

    /// @brief number of messages processed for each channel (ie. next index to send for
    /// each channel)
    std::vector<size_t> per_ch_processed;
};

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * In contrast to `bounded_fanout`, an unbounded fanout supports arbitrary
 * consumption orders of the output channels.
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
    auto& executor = *ctx->executor();

    ShutdownAtExit ch_in_shutdown{ch_in};
    ShutdownAtExit chs_out_shutdown{chs_out};
    co_await executor.schedule();
    UnboundedFanout fanout(chs_out.size());

    std::vector<Node> tasks;
    tasks.reserve(chs_out.size() + 1);

    for (size_t i = 0; i < chs_out.size(); i++) {
        tasks.emplace_back(executor.schedule(
            fanout.send_task(*ctx, fanout.per_ch_processed[i], std::move(chs_out[i]))
        ));
    }
    tasks.emplace_back(executor.schedule(fanout.recv_task(*ctx, std::move(ch_in))));

    coro_results(co_await coro::when_all(std::move(tasks)));
}

}  // namespace

Node fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out,
    FanoutPolicy policy
) {
    RAPIDSMPF_EXPECTS(
        chs_out.size() > 1,
        "fanout requires at least 2 output channels",
        std::invalid_argument
    );

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
