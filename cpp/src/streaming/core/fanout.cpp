/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <deque>
#include <span>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Try to allocate memory from the memory types that the message content uses.
 *
 * @param msg The message to allocate memory for.
 * @return The memory types to try to allocate from.
 */
constexpr std::span<const MemoryType> try_memory_types(Message const& msg) {
    auto const& cd = msg.content_description();
    // if the message content uses device memory, try to allocate from device memory
    // first, else allocate from host memory
    return cd.content_size(MemoryType::DEVICE) > 0
               ? MEMORY_TYPES
               : std::span<const MemoryType>{
                     MEMORY_TYPES.begin() + static_cast<std::size_t>(MemoryType::HOST),
                     MEMORY_TYPES.end()
                 };
}

/**
 * @brief Asynchronously send a message to multiple output channels.
 *
 * @param msg The message to broadcast. Each channel receives a deep copy of the original
 * message.
 * @param chs_out The set of output channels to which the message is sent.
 */
Node send_to_channels(
    Context* ctx, Message&& msg, std::vector<std::shared_ptr<Channel>>& chs_out
) {
    RAPIDSMPF_EXPECTS(!chs_out.empty(), "output channels cannot be empty");

    std::vector<coro::task<bool>> tasks;
    tasks.reserve(chs_out.size());
    for (size_t i = 0; i < chs_out.size() - 1; i++) {
        // do a reservation for each copy, so that it will fallback to host memory if
        // needed
        auto res = ctx->br()->reserve_or_fail(msg.copy_cost(), try_memory_types(msg));
        tasks.emplace_back(chs_out[i]->send(msg.copy(res)));
    }
    // move the message to the last channel to avoid extra copy
    tasks.emplace_back(chs_out.back()->send(std::move(msg)));

    // note that the send tasks may return false if the channel is shut down. But we can
    // safely ignore this in bounded fanout.
    coro_results(co_await coro::when_all(std::move(tasks)));
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
    auto& logger = ctx->logger();
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }

        std::erase_if(chs_out, [](auto&& ch) { return ch->is_shutdown(); });
        if (chs_out.empty()) {
            break;
        }
        co_await send_to_channels(ctx.get(), std::move(msg), chs_out);
        logger.trace("Sent message ", msg.sequence_number());
    }

    std::vector<Node> drain_tasks;
    drain_tasks.reserve(chs_out.size());
    for (auto& ch : chs_out) {
        drain_tasks.emplace_back(ch->drain(ctx->executor()));
    }
    coro_results(co_await coro::when_all(std::move(drain_tasks)));
    logger.trace("Completed bounded fanout");
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
 * - Recv task awaits until the number of cached messages is equal to the latest sent
 * message index by any of the send tasks. This notifies the recv task to pull a message
 * from the input channel, cache it, and notify the send tasks about the new messages.
 * recv task continues this process until the input channel is fully consumed.
 * - Each send task awaits until there are more cached messages to send. Once notified, it
 * determines the current end of the cached messages, and sends messages in the range
 * [next_idx, end_idx). Once these messages have been sent, it updates the next index to
 * end_idx and notifies the recv task.
 *
 * Additional considerations:
 * - In the recv task loop, it also identifies the last completed message index by all
 * send tasks. Message upto this index are no longer needed, and are purged from the
 * cached messages.
 * - When a send task fails to send a message, this means the channel may have been
 * prematurely shut down. In this case, it sets its index to InvalidIdx. Recv task will
 * filter out channels with InvalidIdx.
 * - There two RAII helpers to ensure that the notification mechanisms are properly
 * cleaned up when the unbounded fanout state goes out of scope/ encounters an error.
 *
 */
struct UnboundedFanout {
    /**
     * @brief Constructor.
     *
     * @param num_channels The number of output channels.
     */
    explicit UnboundedFanout(size_t num_channels) : ch_next_idx(num_channels, 0) {}

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
        size_t idx;

        ~SetChannelIdxInvalidAtExit() {
            coro::sync_wait(set_channel_idx_invalid());
        }

        Node set_channel_idx_invalid() {
            if (idx != InvalidIdx) {
                {
                    auto lock = co_await fanout->mtx.scoped_lock();
                    fanout->ch_next_idx[idx] = InvalidIdx;
                }
                co_await fanout->request_data.notify_one();
            }
            idx = InvalidIdx;
        }
    };

    /**
     * @brief Send messages to multiple output channels.
     *
     * @param ctx The context to use.
     * @param self Self index of the task
     * @param ch_out The output channel to send messages to.
     * @return A coroutine representing the task.
     */
    Node send_task(Context& ctx, size_t self, std::shared_ptr<Channel> ch_out) {
        ShutdownAtExit ch_shutdown{ch_out};
        SetChannelIdxInvalidAtExit set_ch_idx_invalid{.fanout = this, .idx = self};
        co_await ctx.executor()->schedule();

        auto& logger = ctx.logger();

        size_t curr_recv_msg_sz = 0;  // current size of the recv_messages deque
        while (true) {
            {
                auto lock = co_await mtx.scoped_lock();
                co_await data_ready.wait(lock, [&] {
                    // irrespective of input_done, update the end_idx to the total number
                    // of messages
                    curr_recv_msg_sz = recv_messages.size();
                    return input_done || ch_next_idx[self] < curr_recv_msg_sz;
                });
                if (input_done && ch_next_idx[self] == curr_recv_msg_sz) {
                    // no more messages will be received, and all messages have been sent
                    break;
                }
            }

            // now we can copy & send messages in indices [next_idx, curr_recv_msg_sz)
            // it is guaranteed that message purging will be done only on indices less
            // than next_idx, so we can safely send messages without locking the mtx
            for (size_t i = ch_next_idx[self]; i < curr_recv_msg_sz; i++) {
                auto const& msg = recv_messages[i];
                RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");

                // make reservations for each message so that it will fallback to host
                // memory if needed
                auto res =
                    ctx.br()->reserve_or_fail(msg.copy_cost(), try_memory_types(msg));
                if (!co_await ch_out->send(msg.copy(res))) {
                    // Failed to send message. Could be that the channel is shut down.
                    // So we need to abort the send task, and notify the process input
                    // task
                    co_await set_ch_idx_invalid.set_channel_idx_invalid();
                    co_return;
                }
            }
            logger.trace(
                "sent ", self, " [", ch_next_idx[self], ", ", curr_recv_msg_sz, ")"
            );

            // now next_idx can be updated to end_idx, and if !input_done, we need to
            // request the recv task for more data
            auto lock = co_await mtx.scoped_lock();
            ch_next_idx[self] = curr_recv_msg_sz;
            if (ch_next_idx[self] == recv_messages.size()) {
                if (input_done) {
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
        logger.trace("Send task ", self, " completed");
    }

    /**
     * @brief RAII helper class to set input_done to true and notify all send tasks to
     * wind down when the unbounded fanout state goes out of scope.
     */
    struct SetInputDoneAtExit {
        UnboundedFanout* fanout;

        ~SetInputDoneAtExit() {
            coro::sync_wait(set_input_done());
        }

        // forcibly set input_done to true and notify all send tasks to wind down
        Node set_input_done() {
            {
                auto lock = co_await fanout->mtx.scoped_lock();
                fanout->input_done = true;
            }
            co_await fanout->data_ready.notify_all();
        }
    };

    /**
     * @brief Wait for a data request from the send tasks.
     *
     * @return The index of the last completed message and the index of the latest
     * processed message. If both are InvalidIdx, it means that all send tasks are in an
     * invalid state.
     */
    auto wait_for_data_request() -> coro::task<std::pair<size_t, size_t>> {
        size_t last_completed_idx = InvalidIdx;
        size_t latest_processed_idx = InvalidIdx;

        auto lock = co_await mtx.scoped_lock();
        co_await request_data.wait(lock, [&] {
            auto filtered_view = std::ranges::filter_view(ch_next_idx, [](size_t idx) {
                return idx != InvalidIdx;
            });

            auto it = std::ranges::begin(filtered_view);
            auto end = std::ranges::end(filtered_view);

            if (it == end) {
                // no valid indices, so all send tasks are in an invalid state
                return true;
            }

            auto [min_it, max_it] = std::minmax_element(it, end);
            last_completed_idx = *min_it;
            latest_processed_idx = *max_it;

            return latest_processed_idx == recv_messages.size();
        });

        co_return std::make_pair(last_completed_idx, latest_processed_idx);
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
        auto& logger = ctx.logger();

        logger.trace("Scheduled process input task");

        // index of the first message to purge
        size_t purge_idx = 0;

        // input_done is only set by this task, so reading without lock is safe here
        while (!input_done) {
            auto [last_completed_idx, latest_processed_idx] =
                co_await wait_for_data_request();
            if (last_completed_idx == InvalidIdx && latest_processed_idx == InvalidIdx) {
                break;  // all send tasks are in an invalid state, so we need to break
            }

            // receive a message from the input channel
            auto msg = co_await ch_in->receive();

            {  // relock mtx to update input_done/recv_messages
                auto lock = co_await mtx.scoped_lock();
                if (msg.empty()) {
                    input_done = true;
                } else {
                    recv_messages.emplace_back(std::move(msg));
                }
            }

            // notify send_tasks to copy & send messages
            co_await data_ready.notify_all();

            // purge completed send_tasks. This will reset the messages to empty, so that
            // they release the memory, however the deque is not resized. This guarantees
            // that the indices are not invalidated. intentionally not locking the mtx
            // here, because we only need to know a lower-bound on the last completed idx
            // (ch_next_idx values are monotonically increasing)
            while (purge_idx + 1 < last_completed_idx) {
                recv_messages[purge_idx].reset();
                purge_idx++;
            }
            logger.trace("recv_messages active size: ", recv_messages.size() - purge_idx);
        }

        co_await ch_in->drain(ctx.executor());
        logger.trace("Process input task completed");
    }

    coro::mutex mtx;  ///< notify send tasks to copy & send messages
    coro::condition_variable
        data_ready;  ///< notify send tasks to copy & send messages notify this task to
                     ///< receive more data from the input channel
    coro::condition_variable
        request_data;  ///< notify recv task to receive more data from the input channel
    bool input_done{false};  ///< set to true when the input channel is fully consumed
    std::deque<Message>
        recv_messages;  ///< messages received from the input channel. We use a deque to
                        ///< avoid references being invalidated by reallocations.
    std::vector<size_t> ch_next_idx;  ///< next index to send for each channel
};

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * This is an all-purpose implementation that can support consuming messages by the
 * channel order or message order. Output channels could be connected to
 * single/multiple consumer nodes. A consumer node can decide to consume all messages
 * from a single channel before moving to the next channel, or it can consume messages
 * from all channels before moving to the next message. When a message has been sent
 * to all output channels, it is purged from the internal deque.
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
    co_await ctx->executor()->schedule();
    auto& logger = ctx->logger();
    auto fanout = std::make_unique<UnboundedFanout>(chs_out.size());

    std::vector<Node> tasks;
    tasks.reserve(chs_out.size() + 1);

    for (size_t i = 0; i < chs_out.size(); i++) {
        tasks.emplace_back(
            executor.schedule(fanout->send_task(*ctx, i, std::move(chs_out[i])))
        );
    }
    tasks.emplace_back(executor.schedule(fanout->recv_task(*ctx, std::move(ch_in))));

    coro_results(co_await coro::when_all(std::move(tasks)));
    logger.debug("Unbounded fanout completed");
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
