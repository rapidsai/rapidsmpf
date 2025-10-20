/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <typeinfo>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Type-erased message wrapper around a payload.
 *
 * Holds a type-erased payload that may be shared with other messages. When the
 * payload is shared, the handle still provides const access via `get<T>()`, but
 * transferring ownership with `release<T>()` is only allowed if the message is
 * solely owned.
 *
 * @note Copying is intentionally restricted; use `shallow_copy()` to create
 * another handle that shares the same payload.
 */
class Message {
    /**
     * @brief Internal container for the (shared) payload.
     */
    struct Payload {
        mutable std::mutex mutex;
        /**
         * @brief Type-erased holder for the actual payload.
         *
         * Stores a `std::shared_ptr<T>` for some `T`.
         *
         * @note Conceptually, a `std::unique_ptr` would suffice for exclusive ownership,
         * but `std::any` requires its contents to be copyable, so a `std::shared_ptr` is
         * used instead.
         */
        std::any data;
    };

  public:
    /// @brief Create an empty message.
    Message() = default;

    /**
     * @brief Construct a message from a unique pointer to a payload.
     *
     * Takes ownership of @p payload and stores it as a shared object internally.
     *
     * @tparam T Payload type.
     * @param payload Non-null unique pointer to the payload instance.
     * @throws std::invalid_argument if @p payload is null.
     */
    template <typename T>
    Message(std::unique_ptr<T> payload) {
        RAPIDSMPF_EXPECTS(
            payload != nullptr, "payload cannot be null", std::invalid_argument
        );
        payload_ = std::make_shared<Payload>();
        payload_->data = std::shared_ptr<T>(std::move(payload));
    }

    /** @brief Move construct. @param other Source message. */
    Message(Message&& other) noexcept = default;

    /** @brief Move assign. @param other Source message. @return *this. */
    Message& operator=(Message&& other) noexcept = default;

    /**
     * @brief Reset the message to the empty state.
     *
     * Releases the stored payload (if this is the last shared owner).
     *
     * @note After this call, the message is empty.
     */
    void reset() noexcept {
        return payload_.reset();
    }

    /**
     * @brief Whether the message currently holds a payload.
     *
     * @return `true` if the message is empty; otherwise, `false`.
     */
    [[nodiscard]] bool empty() const noexcept {
        if (payload_) {
            std::lock_guard<std::mutex> lock(payload_->mutex);
            return !payload_->data.has_value();
        }
        return true;
    }

    /**
     * @brief Check whether the stored payload has type `T`.
     *
     * @tparam T Expected payload type.
     * @return `true` if the payload type matches; otherwise, `false`.
     * @throws std::invalid_argument if the message is empty.
     */
    template <typename T>
    [[nodiscard]] bool holds() const {
        auto lock = lock_payload();
        return payload_->data.type() == typeid(std::shared_ptr<T>);
    }

    /**
     * @brief Access the payload by const reference.
     *
     * The returned reference remains valid until the message is reset or the payload is
     * released.
     *
     * @tparam T Payload type.
     * @return Const reference to the stored payload.
     * @throws std::invalid_argument if the message is empty or the type mismatches.
     */
    template <typename T>
    T const& get() {
        auto [ret, lock] = get_ptr_and_lock<T>();
        return *ret;
    }

    /**
     * @brief Move the payload out of the message and empty it.
     *
     * Transfers ownership of the payload out of the message. This operation is
     * only allowed if the message is the sole owner of the payload.
     *
     * @tparam T Payload type to extract.
     * @return The payload moved out of the message.
     *
     * @throws std::invalid_argument if:
     *         - The message is empty.
     *         - The stored type does not match `T`.
     *         - There are multiple shared references to the same payload.
     *
     * @note After this call, the message is empty.
     */
    template <typename T>
    T release() {
        // If this is the last reference, `reset()` deallocates `payload_` thus
        // we have to extract the payload before resetting.
        auto ret = [&]() -> std::shared_ptr<T> {
            auto [ptr, lock] = get_ptr_and_lock<T>();
            RAPIDSMPF_EXPECTS(
                payload_.use_count() == 1,
                "release() requires this to be the sole owner of the payload",
                std::invalid_argument
            );
            return std::move(ptr);
        }();
        reset();
        return std::move(*ret);
    }

    /**
     * @brief Create a shallow copy that shares the payload.
     *
     * Produces another handle to the same underlying payload; no data is copied.
     *
     * @return A new message that shares the payload with this message.
     */
    [[nodiscard]] Message shallow_copy() const {
        auto lock = lock_payload();
        return *this;
    }

  private:
    // Copying is private to force explicit sharing via shallow_copy().
    Message(Message const&) = default;
    Message& operator=(Message const&) = default;

    /**
     * @brief Lock and validate the internal payload.
     *
     * @return A unique lock that guards the payload mutex.
     * @throws std::invalid_argument if the message is empty.
     */
    [[nodiscard]] std::unique_lock<std::mutex> lock_payload() const {
        if (payload_) {
            std::unique_lock<std::mutex> lock(payload_->mutex);
            if (payload_->data.has_value()) {
                return lock;
            }
        }
        RAPIDSMPF_FAIL("message is empty", std::invalid_argument);
    }

    /**
     * @brief Get a shared pointer to the payload of type `T`, with the payload locked.
     *
     * @tparam T Payload type.
     * @return Shared pointer to the payload and its lock.
     * @throws std::invalid_argument if the message is empty or the type mismatches.
     */
    template <typename T>
    [[nodiscard]] std::pair<std::shared_ptr<T>, std::unique_lock<std::mutex>>
    get_ptr_and_lock() const {
        RAPIDSMPF_EXPECTS(holds<T>(), "wrong message type", std::invalid_argument);
        auto lock = lock_payload();
        return std::make_pair(
            std::any_cast<std::shared_ptr<T>>(payload_->data), std::move(lock)
        );
    }

  private:
    std::shared_ptr<Payload> payload_;
};

}  // namespace rapidsmpf::streaming
