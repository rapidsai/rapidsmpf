/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/core/spillable_messages.hpp>

#include <coro/coro.hpp>
#include <coro/semaphore.hpp>

namespace rapidsmpf::streaming {

class Context;

/**
 * @brief An awaitable semaphore to manage acquisition and release of finite resources.
 */
using Semaphore = coro::semaphore<std::numeric_limits<std::ptrdiff_t>::max()>;

/**
 * @brief A coroutine-based channel for sending and receiving messages asynchronously.
 *
 * The constructor is private, use the factory method `Context::create_channel()` to
 * create a new channel.
 */
class Channel {
    friend Context;

  public:
    /**
     * @brief Asynchronously send a message into the channel.
     *
     * Suspends if the channel is full.
     *
     * @param msg The msg to send.
     * @return A coroutine that evaluates to true if the msg was successfully sent or
     * false if the channel was shut down.
     */
    coro::task<bool> send(Message msg);

    /**
     * @brief Asynchronously receive a message from the channel.
     *
     * Suspends if the channel is empty.
     *
     * @return A coroutine that evaluates to the message, which will be empty if the
     * channel is shut down.
     */
    coro::task<Message> receive();

    /**
     * @brief Drains all pending messages from the channel and shuts it down.
     *
     * This is intended to ensure all remaining messages are processed.
     *
     * @param executor The thread pool used to process remaining messages.
     * @return A coroutine representing the completion of the shutdown drain.
     */
    Node drain(std::unique_ptr<coro::thread_pool>& executor);

    /**
     * @brief Immediately shuts down the channel.
     *
     * Any pending or future send/receive operations will complete with failure.
     *
     * @return A coroutine representing the completion of the shutdown.
     */
    Node shutdown();

    /**
     * @brief Check whether the channel is empty.
     *
     * @return True if there are no messages in the buffer.
     */
    [[nodiscard]] bool empty() const noexcept;

  private:
    Channel(std::shared_ptr<SpillableMessages> spillable_messages)
        : sm_{std::move(spillable_messages)} {}

    coro::ring_buffer<SpillableMessages::MessageId, 1> rb_;
    std::shared_ptr<SpillableMessages> sm_;
};

/**
 * @brief An adaptor to throttle access to a channel.
 *
 * This adds a semaphore-based throttle to a channel to cap the number of suspended
 * coroutines that can be waiting to send into it. It is useful when writing producer
 * nodes that otherwise do not depend on an input channel.
 */
class ThrottlingAdaptor {
  private:
    ///< @brief Ticket with permission to send into the channel.
    class Ticket {
      public:
        Ticket& operator=(Ticket const&) = delete;
        Ticket(Ticket const&) = delete;
        Ticket& operator=(Ticket&&) = default;
        Ticket(Ticket&&) = default;
        ~Ticket() = default;

        /**
         * @brief Asynchronously send a message into the channel.
         *
         * Suspends if the channel is full.
         *
         * @throws std::logic_error If attempting to send more than once with the same
         * ticket.
         *
         * @param msg The msg to send.
         *
         * @return A coroutine that evaluates to a pair of true if the msg was
         * successfully sent or false if the channel was shut down and a release task to
         * be awaited on.
         *
         * @note It is permissible to release a receipt after the channel is shut down,
         * but any waiters will wake in a failed state.
         *
         * @note To avoid immediately transferring control in this executing thread to any
         * waiting tasks, one should `yield` in the executor before awaiting the release
         * task.
         *
         * Here is a typical pattern:
         * @code{.cpp}
         * auto ticket = co_await channel.acquire();
         * auto msg = do_expensive_work();
         * auto [_, receipt] = co_await ticket.send(msg);
         * co_await executor->yield();
         * co_await receipt;
         * @endcode
         *
         * The reason for the `yield` is that when releasing the semaphore, libcoro
         * transfers control directly from the thread executing a `release` to any
         * suspended `acquire` call and keeps executing. Suppose we have `N` threads
         * suspended at `acquire`. If a thread makes it to `release` we typically don't
         * want it to pick up the next suspended acquire task, we want it to park and let
         * someone else pick up a task. By yielding before `release` we introduce a point
         * where we can swap the thread out for another one by moving this coroutine to
         * the back of the queue.
         */
        [[nodiscard]] coro::task<std::pair<bool, coro::task<void>>> send(Message msg) {
            RAPIDSMPF_EXPECTS(ch_, "Ticket has already been used", std::logic_error);
            auto sent = co_await ch_->send(std::move(msg));
            ch_ = nullptr;
            if (sent) {
                co_return {sent, semaphore_->release()};
            } else {
                // If the channel is closed we want to wake any waiters so shutdown the
                // semaphore.
                co_await semaphore_->shutdown();
                co_return {sent, []() -> coro::task<void> { co_return; }()};
            }
        }

        /**
         * @brief Create a ticket permitting a send into a channel.
         *
         * @param channel The channel to send into.
         * @param semaphore Semaphore to release after send is complete.
         */
        Ticket(Channel* channel, Semaphore* semaphore)
            : ch_{channel}, semaphore_{semaphore} {};

      private:
        Channel* ch_;
        Semaphore* semaphore_;
    };

  public:
    /**
     * @brief Create an adaptor that throttles sends into a channel.
     *
     * @param channel Channel to throttle.
     * @param max_tickets Maximum number of simultaneous tickets for sending into the
     * channel.
     *
     * This adaptor is typically used for producer tasks that have no dependencies but
     * where we nonetheless want to introduce a suspension point before sending into an
     * output channel. Such a task can accept the output channel and wrap it in a
     * `ThrottlingAdaptor`. Consumers of the adapted channel must first acquire a ticket
     * to send before they can send. At most `max_tickets` consumers can pass the acquire
     * suspension point at once.
     *
     * Example usage:
     * @code{.cpp}
     * auto ch = ctx->create_channel();
     * auto throttled = ThrottlingAdaptor(ch, 4);
     * auto make_task = [&]() {
     *     auto ticket = co_await throttled.acquire();
     *     auto data = do_expensive_work();
     *     auto [_, receipt] = co_await ticket.send(data);
     *     // Not for correctness, but to allow other threads to pick up awaiters at
     *     // acquire
     *     co_await executor->yield();
     *     co_await receipt;
     * };
     * std::vector<coro::task<void>> tasks;
     * for ( ... ) {
     *     tasks.push_back(make_task());
     * }
     * co_await coro::when_all(std::move(tasks));
     * @endcode
     */
    explicit ThrottlingAdaptor(
        std::shared_ptr<Channel> channel, std::ptrdiff_t max_tickets
    )
        : ch_{std::move(channel)}, semaphore_(max_tickets) {
        RAPIDSMPF_EXPECTS(
            max_tickets > 0, "ThrottlingAdaptor must have at least one ticket"
        );
    }

    /**
     * @brief Obtain a ticket to send a message.
     *
     * Suspends if all tickets are currently handed out.
     *
     * @throws std::runtime_error If the semaphore is shut down.
     *
     * @return A coroutine producing a new `Ticket` that grants permission to send a
     * message.
     */
    [[nodiscard]] coro::task<Ticket> acquire() {
        auto result = co_await semaphore_.acquire();
        RAPIDSMPF_EXPECTS(
            result == coro::semaphore_acquire_result::acquired,
            "Semaphore was shutdown",
            std::runtime_error
        );
        co_return {ch_.get(), &semaphore_};
    }

  private:
    std::shared_ptr<Channel> ch_;
    Semaphore semaphore_;
};

/**
 * @brief Helper RAII class to shut down channels when they go out of scope.
 *
 * When this object is destroyed, it invokes `shutdown()` on all provided channels
 * in the order they were provided. After shutdown, any pending or future send/receive
 * operations on those channels will fail or yield `nullopt`.
 *
 * This is useful inside coroutine bodies to guarantee channels are shut down if an
 * unhandled exception escapes the coroutine. Relying on a channel's own destructor
 * is insufficient when the channel is shared (e.g., via `std::shared_ptr`), because
 * other owners keep it alive.
 */
class ShutdownAtExit {
  public:
    /**
     * @brief Construct from a vector of channel handles.
     *
     * The order of elements determines the shutdown order invoked by the destructor.
     *
     * @param channels Vector of shared channel handles to be shut down on destruction.
     *
     * @throws std::invalid_argument If any channel in the vector is `nullptr`.
     */
    explicit ShutdownAtExit(std::vector<std::shared_ptr<Channel>> channels)
        : channels_{std::move(channels)} {
        for (auto& ch : channels_) {
            RAPIDSMPF_EXPECTS(ch, "channel cannot be null", std::invalid_argument);
        }
    }

    /**
     * @brief Variadic convenience constructor.
     *
     * Enables `ShutdownAtExit{ch1, ch2, ...}` without explicitly creating a vector.
     * Each argument must be convertible to `std::shared_ptr<Channel>`. The order of
     * the arguments determines the shutdown order in the destructor.
     *
     * @tparam T Parameter pack of types convertible to `std::shared_ptr<Channel>`.
     * @param channels One or more channel handles.
     *
     * @throws std::invalid_argument If any of the provided channel pointers is `nullptr`.
     */
    template <class... T>
    explicit ShutdownAtExit(T&&... channels)
        requires(std::convertible_to<T, std::shared_ptr<Channel>> && ...)
        : ShutdownAtExit(
              std::vector<std::shared_ptr<Channel>>{std::forward<T>(channels)...}
          ) {}

    // Non-copyable, non-movable.
    ShutdownAtExit(ShutdownAtExit const&) = delete;
    ShutdownAtExit& operator=(ShutdownAtExit const&) = delete;
    ShutdownAtExit(ShutdownAtExit&&) = delete;
    ShutdownAtExit& operator=(ShutdownAtExit&&) = delete;

    /**
     * @brief Destructor that synchronously shuts down all channels.
     *
     * Calls `shutdown()` on each channel in the same order they were passed.
     */
    ~ShutdownAtExit() noexcept {
        for (auto& ch : channels_) {
            coro::sync_wait(ch->shutdown());
        }
    }

  private:
    std::vector<std::shared_ptr<Channel>> channels_;
};

}  // namespace rapidsmpf::streaming
