/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include <coro/queue.hpp>
#include <coro/sync_wait.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief A bounded queue for type-erased `Message`s
 *
 * This adds a semaphore-based ticketing system to `coro::queue`. The producer must
 * acquire a ticket which is sent with the message to the consumer who can decide when to
 * release the ticket having received the message.
 */
class BoundedQueue {
    friend Context;

  private:
    /**
     * @brief Helper class to shutdown the queue at exit of a function via RAII.
     */
    class Shutdown {
      public:
        /**
         * @brief Construct from a queue reference
         *
         * @param q the queue to shut down.
         */
        explicit Shutdown(BoundedQueue& q) : q_{q} {}

        /**
         * @brief Destructor that synchronously shuts down the contained queue.
         */
        ~Shutdown() noexcept {
            coro::sync_wait(q_.shutdown());
        }

      private:
        BoundedQueue& q_;
    };

    ///< @brief Ticket with permission to send into the queue.
    class Ticket {
      public:
        Ticket& operator=(Ticket const&) = delete;
        Ticket(Ticket const&) = delete;
        Ticket& operator=(Ticket&&) = default;
        Ticket(Ticket&&) = default;
        ~Ticket() = default;

        /**
         * @brief Create a ticket permitting a send into a queue.
         *
         * @param q The queue to send into.
         * @param semaphore Semaphore to release after send is complete.
         */
        Ticket(coro::queue<Message>* q, Semaphore* semaphore)
            : q_{q}, semaphore_{semaphore} {}

        /**
         * @brief Asynchronously send a message into the queue.
         *
         * @throws std::logic_error If attempting to send more than once with the same
         * ticket.
         *
         * @param msg The message to send.
         *
         * @return A coroutine that evaluates to true if the send was successful and false
         * if the queue was shut down.
         *
         * @note the receiver on the other end of the queue receives a pair of a release
         * task (to be awaited) and the message.
         */
        [[nodiscard]] coro::task<bool> send(Message msg) {
            RAPIDSMPF_EXPECTS(
                q_ && semaphore_, "Ticket has already been used", std::logic_error
            );
            auto result = co_await q_->push(std::move(msg));
            q_ = nullptr;
            semaphore_ = nullptr;
            co_return result == coro::queue_produce_result::produced;
        }

      private:
        coro::queue<Message>* q_;
        Semaphore* semaphore_;
    };

    /**
     * @brief A queue with a maximum capacity, managed by a semaphore.
     *
     * @param buffer_size Maximum capacity in the queue.
     *
     * @note In contrast to a simple bounded queue, we split the acquisition of
     * permission to send in to the queue with the send itself.
     *
     * Hence the pattern is first to acquire the resource, then compute the message,
     * followed by a send of the message. The consumer then releases the resource when
     * they are ready.
     */
    BoundedQueue(std::size_t buffer_size)
        : semaphore_{static_cast<std::ptrdiff_t>(buffer_size)} {};

  public:
    /**
     * @brief Acquire a ticket to send into the queue.
     *
     * @return A coroutine to be awaited that provides a ticket (or `std::nullopt` if the
     * queue is shutdown).
     */
    [[nodiscard]] coro::task<std::optional<Ticket>> acquire() {
        auto result = co_await semaphore_.acquire();
        if (result == coro::semaphore_acquire_result::shutdown) {
            co_return std::nullopt;
        }
        co_return std::make_optional<Ticket>(&q_, &semaphore_);
    }

    /**
     * @brief Receive a message from the queue.
     *
     * @return A coroutine containing the release task and the received message (or a null
     * task and an empty message if the queue is shut down).
     */
    [[nodiscard]] coro::task<std::pair<coro::task<void>, Message>> receive() {
        auto msg = co_await q_.pop();
        if (msg.has_value()) {
            co_return {semaphore_.release(), std::move(*msg)};
        } else {
            co_return {[]() -> coro::task<void> { co_return; }(), Message{}};
        }
    }

    /**
     * @brief Drain all messages in the queue and shut down.
     *
     * @param executor The thread pool used to process the remaining messages.
     *
     * @return A coroutine representing completion of the shutdown drain.
     */
    [[nodiscard]] coro::task<void> drain(std::unique_ptr<coro::thread_pool>& executor) {
        co_await q_.shutdown_drain(executor);
        co_await semaphore_.shutdown();
    }

    /**
     * @brief Immediately shut down the queue.
     *
     * Any pending or future operations will complete with failure.
     *
     * @return A coroutine representing the shutdown.
     */
    [[nodiscard]] coro::task<void> shutdown() {
        coro_results(co_await coro::when_all(q_.shutdown(), semaphore_.shutdown()));
    }

    /**
     * @brief Obtain an object that will synchronously shutdown the queue when it goes out
     * of scope.
     *
     * @return A shutdown object.
     */
    [[nodiscard]] BoundedQueue::Shutdown raii_shutdown() noexcept {
        return BoundedQueue::Shutdown{*this};
    }

  private:
    coro::queue<Message> q_{};  ///< Queue for messages
    Semaphore semaphore_;  ///< Semaphore managing size of queue.
};

}  // namespace rapidsmpf::streaming
