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
    /**
     * @brief Callback functions associated with a message.
     *
     * Allows type-specific customization of memory-related operations, such as
     * computing data sizes or performing deep copies.
     */
    struct Callbacks {
        /**
         * @brief Callback for computing the size of a message's primary data.
         *
         * This callback returns the total size, in bytes, of the data portion associated
         * with a message. It is used to determine memory requirements for the primary
         * data only — not for metadata or any auxiliary information.
         *
         * Typically, this data is represented by a `Buffer` that may reside in any
         * memory type (e.g., host or device). The size reported by this callback
         * determines how large a memory reservation must be when performing a
         * `Message::copy()`.
         *
         * @param msg Reference to the message whose data size is queried.
         * @param mem_type Target memory type to query.
         * @return A pair (size, spillable) where:
         *   - size: total size of the primary data for the given memory type.
         *   - spillable: `true` if the message owns its buffers and destroying it will
         *     release the associated memory; otherwise `false`.
         */
        std::function<std::pair<size_t, bool>(Message const&, MemoryType)>
            primary_data_size;

        /**
         * @brief Callback for performing a deep copy of a message.
         *
         * The copy operation allocates new memory for the payload using the provided
         * buffer resource and memory reservation. The memory type specified in the
         * reservation determines where the new copy will reside (e.g., device or host
         * memory).
         *
         * @param msg Source message to copy.
         * @param br Buffer resource used for allocations.
         * @param reservation Memory reservation to consume.
         *
         * @return A new `Message` instance containing a deep copy of the payload.
         */
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

    /**
     * @brief Returns the callbacks associated with this message.
     *
     * The callbacks define custom behaviors for operations such as
     * `primary_data_size()` and `copy()`.
     *
     * @return Constant reference to the message's registered callbacks.
     */
    [[nodiscard]] Callbacks const& callbacks() const noexcept {
        return callbacks_;
    }

    [[nodiscard]] std::pair<size_t, bool> primary_data_size(MemoryType mem_type) {
        RAPIDSMPF_EXPECTS(
            callbacks_.primary_data_size,
            "message doesn't support `primary_data_size`",
            std::invalid_argument
        );
        return callbacks_.primary_data_size(*this, mem_type);
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
