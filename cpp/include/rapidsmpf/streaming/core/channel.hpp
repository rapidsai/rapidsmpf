/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <any>
#include <cstddef>
#include <memory>
#include <typeinfo>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include <coro/coro.hpp>
#include <coro/semaphore.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Move-only, type-erased message holding a payload as shared pointer.
 */
class Message {
  public:
    Message() = default;

    /**
     * @brief Construct from a unique pointer (promoted to shared_ptr).
     *
     * @tparam T Payload type.
     * @param ptr Non-null unique pointer.
     * @throws std::invalid_argument if @p ptr is null.
     */
    template <typename T>
    Message(std::unique_ptr<T> ptr) {
        RAPIDSMPF_EXPECTS(ptr != nullptr, "nullptr not allowed", std::invalid_argument);
        data_ = std::shared_ptr<T>(std::move(ptr));
    }

    /** @brief Move construct. @param other Source message. */
    Message(Message&& other) noexcept = default;

    /** @brief Move assign. @param other Source message. @return *this. */
    Message& operator=(Message&& other) noexcept = default;
    Message(Message const&) = delete;
    Message& operator=(Message const&) = delete;

    /**
     * @brief Reset the message to empty.
     */
    void reset() noexcept {
        return data_.reset();
    }

    /**
     * @brief Returns true when no payload is stored.
     *
     * @return true if empty, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept {
        return !data_.has_value();
    }

    /**
     * @brief Compare the payload type.
     *
     * @tparam T Expected payload type.
     * @return true if the payload is `typeid(T)`, false otherwise.
     */
    template <typename T>
    [[nodiscard]] bool holds() const noexcept {
        return data_.type() == typeid(std::shared_ptr<T>);
    }

    /**
     * @brief Extracts the payload and resets the message.
     *
     * @tparam T Payload type.
     * @return The payload.
     * @throws std::invalid_argument if empty or type mismatch.
     */
    template <typename T>
    T release() {
        auto ret = get_ptr<T>();
        reset();
        return std::move(*ret);
    }

    /**
     * @brief Reference to the payload.
     *
     * The returned reference remains valid until the message is released or reset.
     *
     * @tparam T Payload type.
     * @return Reference to the payload.
     * @throws std::invalid_argument if empty or type mismatch.
     */
    template <typename T>
    T const& get() {
        return *get_ptr<T>();
    }

  private:
    /**
     * @brief Returns a shared pointer to the payload.
     *
     * @tparam T Payload type.
     * @return std::shared_ptr<T> to the payload.
     * @throws std::invalid_argument if empty or type mismatch.
     */
    template <typename T>
    [[nodiscard]] std::shared_ptr<T> get_ptr() const {
        RAPIDSMPF_EXPECTS(!empty(), "message is empty", std::invalid_argument);
        RAPIDSMPF_EXPECTS(holds<T>(), "wrong message type", std::invalid_argument);
        return std::any_cast<std::shared_ptr<T>>(data_);
    }

  private:
    std::any data_;
};

/**
 * @brief A coroutine-based channel for sending and receiving messages asynchronously.
 */
class Channel {
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
    coro::task<bool> send(Message msg) {
        auto result = co_await rb_.produce(std::move(msg));
        co_return result == coro::ring_buffer_result::produce::produced;
    }

    /**
     * @brief Asynchronously receive a message from the channel.
     *
     * Suspends if the channel is empty.
     *
     * @return A coroutine that evaluates to the message, which will be empty if the
     * channel is shut down.
     */
    coro::task<Message> receive() {
        auto msg = co_await rb_.consume();
        if (msg.has_value()) {
            co_return std::move(*msg);
        } else {
            co_return Message{};
        }
    }

    /**
     * @brief Drains all pending messages from the channel and shuts it down.
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
     * Any pending or future send/receive operations will complete with failure.
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
    coro::ring_buffer<Message, 1> rb_;
};

/**
 * @brief A coroutine-based channel for sending and receiving messages asynchronously.
 *
 * This channel is throttled to cap the number of suspended coroutines that can be waiting
 * to send into it. It is useful when writing producer nodes that
 * otherwise do not depend on an input channel.
 */
class ThrottledChannel {
  private:
    ///< @brief Ticket granting permission to send into the channel.
    class Ticket {
        friend class ThrottledChannel;

      public:
        Ticket& operator=(Ticket const&) = delete;
        Ticket(Ticket const&) = delete;
        Ticket& operator=(Ticket&&) = default;
        Ticket(Ticket&&) = default;
        ~Ticket() = default;

      private:
        Ticket() = default;
    };

    ///< @brief Receipt proving that a ticket has been consumed and sent.
    class Receipt {
        friend class ThrottledChannel;

      public:
        Receipt& operator=(Receipt const&) = delete;
        Receipt(Receipt const&) = delete;
        Receipt& operator=(Receipt&&) = default;
        Receipt(Receipt&&) = default;
        ~Receipt() = default;

      private:
        Receipt() = default;
    };

  public:
    /**
     * @brief Create a new `ThrottledChannel`
     *
     * @param max_tickets Maximum number of simultaneous tickets for sending into the
     * channel.
     */
    explicit ThrottledChannel(std::ptrdiff_t max_tickets)
        : rb_{}, semaphore_(max_tickets) {}

    /**
     * @brief Asynchronously send a message into the channel.
     *
     * Suspends if the channel is full.
     *
     * @param msg The msg to send.
     * @param ticket The ticket indicating one has a right to send.
     *
     * @note One should `release` the ticket after sending into the
     * channel using the receipt provided.
     *
     * @return A coroutine that evaluates to a pair of true if the msg was successfully
     * sent or false if the channel was shut down and a `SentTicket` to be released.
     */
    [[nodiscard]] coro::task<std::pair<bool, Receipt>> send(
        Message msg, [[maybe_unused]] Ticket&& ticket
    ) {
        auto result = co_await rb_.produce(std::move(msg));
        co_return {result == coro::ring_buffer_result::produce::produced, Receipt{}};
    }

    /**
     * @brief Obtain a ticket to send a message.
     *
     * Suspends if all tickets are currently handed out.
     *
     * @return A coroutine producing a new ticket to be used in `send`.
     */
    [[nodiscard]] coro::task<Ticket> acquire() {
        auto result = co_await semaphore_.acquire();
        RAPIDSMPF_EXPECTS(
            result == coro::semaphore_acquire_result::acquired, "Semaphore was shutdown"
        );
        co_return Ticket{};
    }

    /**
     * @brief Release a ticket that has been converted to a sent message.
     *
     * @param receipt The receipt indicating one has sent a message.
     * @return A coroutine representing completion of the release.

     * @note To avoid immediately transferring control in this
     * executing thread to any waiting tasks, one should `yield` in
     * the executor before `release`ing.
     *
     * Here is a typical pattern:
     * @code{.cpp}
     * auto ticket = co_await channel.acquire();
     * auto data = do_expensive_work();
     * auto [_, receipt] = co_await channel.send(std::move(ticket));
     * co_await executor->yield();
     * co_await channel.release(std::move(receipt));
     * @endcode
     */
    coro::task<void> release([[maybe_unused]] Receipt&& receipt) {
        RAPIDSMPF_EXPECTS(!semaphore_.is_shutdown(), "Semaphore was shutdown");
        co_await semaphore_.release();
    }

    /**
     * @brief Asynchronously receive a message from the channel.
     *
     * Suspends if the channel is empty.
     *
     * @return A coroutine that evaluates to the message, which will be empty if the
     * channel is shut down.
     */
    coro::task<Message> receive() {
        auto msg = co_await rb_.consume();
        if (msg.has_value()) {
            co_return std::move(*msg);
        } else {
            co_return Message{};
        }
    }

    /**
     * @brief Drains all pending messages from the channel and shuts it down.
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
     * Any pending or future send/receive operations will complete with failure.
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
    coro::ring_buffer<Message, 1> rb_;
    coro::semaphore<std::numeric_limits<std::ptrdiff_t>::max()> semaphore_;
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
 *
 * @tparam T Variadic list of channel handle types.
 */
template <typename... T>
class ShutdownAtExit {
  public:
    /**
     * @brief Constructor accepting one or more channel handles.
     *
     * @param channels Channels to be shut down on destruction.
     */
    explicit ShutdownAtExit(T... channels) : channels_{channels...} {}

    /**
     * @brief Destructor that synchronously shuts down all channels.
     *
     * Calls `shutdown()` on each channel in the same order they were passed.
     */
    ~ShutdownAtExit() noexcept {
        std::apply(
            [](auto&... ch) { (coro::sync_wait(ch->shutdown()), ...); }, channels_
        );
    }

  private:
    std::tuple<T...> channels_;
};

}  // namespace rapidsmpf::streaming
