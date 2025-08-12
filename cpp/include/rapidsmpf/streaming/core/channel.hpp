/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <optional>

#include <rapidsmpf/streaming/core/node.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief A coroutine-based channel for sending and receiving messages asynchronously.
 *
 * The default capacity is 1, making it suitable for CSP-style rendezvous communication.
 *
 * @tparam T Type of the message passed through the channel.
 * @tparam Capacity Maximum number of messages the channel can buffer.
 */
template <typename T, std::size_t Capacity = 1>
class Channel {
  public:
    /**
     * @brief Asynchronously send a value into the channel.
     *
     * Suspends if the buffer is full. Returns false if the channel is shut down.
     *
     * @param value The value to send (moved).
     * @return A coroutine that evaluates to true if the value was successfully sent.
     */
    coro::task<bool> send(T&& value) {
        auto result = co_await rb_.produce(std::move(value));
        co_return result == coro::ring_buffer_result::produce::produced;
    }

    /**
     * @brief Asynchronously receive a value from the channel.
     *
     * Suspends if the buffer is empty. Returns std::nullopt if the channel is shut down.
     *
     * @return A coroutine that evaluates to an optional containing the received value.
     */
    coro::task<std::optional<T>> receive() {
        auto msg = co_await rb_.consume();
        if (msg.has_value()) {
            co_return std::make_optional(std::move(*msg));
        } else {
            co_return std::nullopt;
        }
    }

    /**
     * @brief Asynchronously receive a value, or return a default if the channel is
     * closed.
     *
     * Useful for fallback behavior when shutdown might occur.
     *
     * @param default_value The value to return if the channel is closed (moved).
     * @return A coroutine that evaluates to the received value or the default.
     */
    coro::task<T> receive_or(T default_value) {
        auto msg = co_await rb_.consume();
        if (msg.has_value()) {
            co_return std::move(*msg);
        } else {
            co_return default_value;
        }
    }

    /**
     * @brief Drains all pending items from the channel and shuts it down.
     *
     * This is intended to ensure all remaining messages are processed.
     *
     * @param executor The thread pool used to process remaining messages.
     * @return A coroutine representing the completion of the shutdown drain.
     */
    Node drain(std::shared_ptr<coro::thread_pool> executor) {
        return rb_.shutdown_drain(std::move(executor));
    }

    /**
     * @brief Immediately shuts down the channel.
     *
     * Any pending or future send/receive operations will complete with failure/nullopt.
     *
     * @return A coroutine representing the completion of the shutdown.
     */
    Node shutdown() {
        return rb_.shutdown();
    }

    /**
     * @brief Check whether the channel is empty.
     *
     * @return True if there are no messages in the buffer.
     */
    [[nodiscard]] bool empty() const noexcept {
        return rb_.empty();
    }

  private:
    coro::ring_buffer<T, Capacity> rb_;
};

/**
 * @brief Type alias for a shared pointer to a channel of unique pointers.
 *
 * This alias is used throughout rapidsmpf.
 *
 * @tparam T The type held inside the shared pointer.
 */
template <typename T>
using SharedChannel = std::shared_ptr<Channel<std::unique_ptr<T>>>;

/**
 * @brief Creates a new shared channel for shared pointer values.
 *
 * @tparam T The type held inside the shared pointer.
 * @return A shared pointer to a new Channel instance.
 */
template <typename T>
std::shared_ptr<Channel<std::unique_ptr<T>>> make_shared_channel() {
    return std::make_shared<Channel<std::unique_ptr<T>>>();
}

/**
 * @brief Helper RAII class to shut down channels when they go out of scope.
 *
 * Automatically shuts down all provided channels in reverse construction order.
 * Any pending or future send/receive operations will complete with failure/nullopt.
 *
 * @tparam T Variadic list of channel pointer types.
 */
template <typename... T>
class ShutdownAtExit {
  public:
    /**
     * @brief Constructor accepting one or more channel shared pointers.
     *
     * @param channels Channels to be shut down on destruction.
     */
    ShutdownAtExit(T... channels) : channels_{channels...} {}

    /**
     * @brief Destructor that synchronously shuts down all channels.
     *
     * Ensures no dangling operations remain when exiting the scope.
     */
    ~ShutdownAtExit() {
        std::apply(
            [](auto&... ch) { (coro::sync_wait(ch->shutdown()), ...); }, channels_
        );
    }

  private:
    std::tuple<T...> channels_;
};

}  // namespace rapidsmpf::streaming
