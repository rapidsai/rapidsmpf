/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <utility>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::streaming {


/**
 * @brief Type-erased message wrapper around a payload.
 */
class Message {
  public:
    struct Callbacks {
        std::function<std::pair<size_t, bool>(Message const&, MemoryType)> buffer_size;
        std::function<
            Message(Message const&, BufferResource* br, MemoryReservation& reservation)>
            copy;
    };

    // @brief Create an empty message.
    Message() = default;

    /**
     * @brief Construct a new message from an unique pointer to the payload.
     *
     * @tparam T Payload type.
     * @param payload Non-null unique pointer to the payload.
     *
     * @throws std::invalid_argument if @p payload is null.
     */
    template <typename T>
    Message(std::unique_ptr<T> payload, Callbacks callbacks = Callbacks{})
        : callbacks_{std::move(callbacks)} {
        RAPIDSMPF_EXPECTS(
            payload != nullptr, "nullptr not allowed", std::invalid_argument
        );
        // Conceptually, a `std::unique_ptr` would be sufficient for exclusive ownership.
        // However, since `std::any` requires its contents to be copyable, we store the
        // payload in a `std::shared_ptr` instead. This shared ownership is internal only
        // and never exposed to the user.
        payload_ = std::shared_ptr<T>(std::move(payload));
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
        return payload_.reset();
    }

    /**
     * @brief Returns true when no payload is stored.
     *
     * @return true if empty, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept {
        return !payload_.has_value();
    }

    /**
     * @brief Compare the payload type.
     *
     * @tparam T Expected payload type.
     * @return true if the payload is `typeid(T)`, false otherwise.
     */
    template <typename T>
    [[nodiscard]] bool holds() const noexcept {
        return payload_.type() == typeid(std::shared_ptr<T>);
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
    T const& get() const {
        return *get_ptr<T>();
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
        std::shared_ptr<T> ret = get_ptr<T>();
        reset();
        return std::move(*ret);
    }

    [[nodiscard]] Callbacks const& callbacks() const noexcept {
        return callbacks_;
    }

    [[nodiscard]] std::pair<size_t, bool> buffer_size(MemoryType mem_type) {
        RAPIDSMPF_EXPECTS(
            callbacks_.buffer_size,
            "message doesn't support `buffer_size`",
            std::invalid_argument
        );
        return callbacks_.buffer_size(*this, mem_type);
    }

    [[nodiscard]] Message copy(BufferResource* br, MemoryReservation& reservation) const {
        RAPIDSMPF_EXPECTS(
            callbacks_.copy, "message doesn't support `copy`", std::invalid_argument
        );
        return callbacks_.copy(*this, br, reservation);
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
        return std::any_cast<std::shared_ptr<T>>(payload_);
    }

  private:
    std::any payload_;
    Callbacks callbacks_;
};

}  // namespace rapidsmpf::streaming
