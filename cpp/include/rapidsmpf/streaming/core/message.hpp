/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/content_description.hpp>

namespace rapidsmpf::streaming {


/**
 * @brief Type-erased message wrapper around a payload.
 */
class Message {
  public:
    /**
     * @brief Callback for performing a deep copy of a message.
     *
     * The copy operation allocates new memory for the message's payload using the
     * provided memory reservation. The memory type specified in the reservation
     * determines where the new copy will primarily reside (e.g., device or host
     * memory).
     *
     * @param msg Source message to copy.
     * @param reservation Memory reservation to consume during allocation.
     * @return A new `Message` instance containing a deep copy of the payload.
     */
    using CopyCallback =
        std::function<Message(Message const&, MemoryReservation& reservation)>;

    /**
     * @brief Callback for moving a message into the memory of a reservation.
     *
     * Unlike `CopyCallback`, the source message is consumed, so its memory is released
     * once the returned message has been built.
     *
     * @param msg Source message to move from.
     * @param reservation Memory reservation to consume during allocation.
     * @return A new `Message` instance holding the payload in the reserved memory.
     */
    using MoveCallback =
        std::function<Message(Message&&, MemoryReservation& reservation)>;

    /**
     * @brief The callbacks a message supports.
     *
     * Either may be null, meaning the message does not support that operation.
     */
    struct Callbacks {
        CopyCallback copy;  ///< Deep copies the message, see `copy()`.
        MoveCallback move;  ///< Moves the message, see `move()`.
    };

    /// @brief Create an empty message.
    Message() = default;

    /**
     * @brief Construct a new message from a unique pointer to its payload.
     *
     * The message may optionally support copy and move operations through
     * user-provided callbacks. An operation whose callback is null is disabled.
     *
     * @tparam T Type of the payload to store inside the message.
     * @param sequence_number Ordering identifier for the message.
     * @param payload Non-null unique pointer to the payload.
     * @param content_description Description of the payload's content. When a
     * callback is provided, this description must accurately reflect the content of the
     * payload (e.g., per-memory-type sizes and spillable status).
     * @param callbacks The copy and move callbacks.
     *
     * @note Sequence numbers are used to ensure that when multiple producers send into
     * the same output channel, channel ordering is preserved. Specifically, the guarantee
     * is that `Channel`s always produce elements in increasing sequence number order. To
     * ensure this, single producers must promise to send into the channels in strictly
     * increasing sequence number order. Behaviour is undefined if not. To ensure
     * insertion into an output channel from multiple producers obeys this invariant, use
     * a `Lineariser`.
     * This promise allows consumers to ensure ordering by buffering at most
     * `num_consumers` messages, rather than needing to buffer the entire channel input.
     *
     * @throws std::invalid_argument if @p payload is null.
     */
    template <typename T>
    Message(
        std::uint64_t sequence_number,
        std::unique_ptr<T> payload,
        ContentDescription content_description,
        Callbacks callbacks
    )
        : sequence_number_(sequence_number),
          content_description_{content_description},
          callbacks_{std::move(callbacks)} {
        RAPIDSMPF_EXPECTS(
            payload != nullptr, "nullptr not allowed", std::invalid_argument
        );
        // Conceptually, a `std::unique_ptr` would be sufficient for exclusive ownership.
        // However, since `std::any` requires its contents to be copyable, we store the
        // payload in a `std::shared_ptr` instead. This shared ownership is internal only
        // and never exposed to the user.
        payload_ = std::shared_ptr<T>(std::move(payload));
    }

    /**
     * @brief Construct a new message that optionally supports copying.
     *
     * Kept for backward compatibility. Prefer the overload taking `Callbacks`.
     *
     * @tparam T Type of the payload to store inside the message.
     * @param sequence_number Ordering identifier for the message.
     * @param payload Non-null unique pointer to the payload.
     * @param content_description Description of the payload's content. When a copy
     * callback is provided, this description must accurately reflect the content of the
     * payload (e.g., per-memory-type sizes and spillable status).
     * @param copy_cb Optional callback used to perform deep copies of the message. If
     * `nullptr`, copying and moving are disabled.
     *
     * @note See the overload taking `Callbacks` for the ordering guarantee sequence
     * numbers provide.
     *
     * @throws std::invalid_argument if @p payload is null.
     */
    template <typename T>
    Message(
        std::uint64_t sequence_number,
        std::unique_ptr<T> payload,
        ContentDescription content_description,
        CopyCallback copy_cb = nullptr
    )
        : Message(
              sequence_number,
              std::move(payload),
              content_description,
              Callbacks{.copy = std::move(copy_cb), .move = nullptr}
          ) {}

    // In tandem with coro::queue the move assignment of std::any breaks GCC's
    // uninitialized variable tracking and we get a warning that std::any::_M_manager' may
    // be used uninitialized [-Wmaybe-uninitialized] This is a bug in GCC 14.x which we
    // workaround by suppressing the warning for the move ctors/assignment. Fixed in
    // GCC 15.2.
#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    /** @brief Move construct. @param other Source message. */
    Message(Message&& other) noexcept = default;

    /** @brief Move assign. @param other Source message. @return *this. */
    Message& operator=(Message&& other) noexcept = default;
#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic pop
#endif
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
     * @brief Returns the sequence number of this message.
     *
     * @return The sequence number.
     */
    [[nodiscard]] constexpr std::uint64_t sequence_number() const noexcept {
        return sequence_number_;
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
    [[nodiscard]] T const& get() const {
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
    [[nodiscard]] T release() {
        std::shared_ptr<T> ret = get_ptr<T>();
        reset();
        return std::move(*ret);
    }

    /**
     * @brief Returns the content description associated with the message.
     *
     * @return The message's content description.
     */
    [[nodiscard]] constexpr ContentDescription const&
    content_description() const noexcept {
        return content_description_;
    }

    /**
     * @brief Returns the callbacks associated with the message.
     *
     * @return The message's copy and move callbacks.
     */
    [[nodiscard]] constexpr Callbacks const& callbacks() const noexcept {
        return callbacks_;
    }

    /**
     * @brief Returns the copy callback associated with the message.
     *
     * Kept for backward compatibility, prefer `callbacks()`.
     *
     * @return The message's copy callback function.
     */
    [[nodiscard]] constexpr CopyCallback const& copy_cb() const noexcept {
        return callbacks().copy;
    }

    /**
     * @brief Returns the total memory size required for a deep copy of the payload.
     *
     * The computed size represents the total amount of memory that must be
     * reserved to duplicate all content buffers of the message, regardless of
     * their current memory locations. For example, if the payload's content
     * resides in both host and device memory, the returned size is the sum of
     * both.
     *
     * @return Total number of bytes that must be reserved to perform a deep copy
     * of the message's payload and content buffers.
     *
     * @see copy()
     */
    [[nodiscard]] constexpr std::size_t copy_cost() const noexcept {
        return content_description().content_size();
    }

    /**
     * @brief Perform a deep copy of this message and its payload.
     *
     * Invokes the registered `copy` callback to create a new `Message` with freshly
     * allocated buffers. The allocation is performed using the provided memory
     * reservation, which also define the target memory type (e.g., host or device).
     *
     * The resulting message contains a deep copy of the original payload, while
     * preserving the same metadata and callbacks.
     *
     * @param reservation Memory reservation to consume for the copy.
     * @return A new `Message` instance containing a deep copy of the payload.
     *
     * @throws std::invalid_argument if the message does not support copying.
     */
    [[nodiscard]] Message copy(MemoryReservation& reservation) const {
        RAPIDSMPF_EXPECTS(
            callbacks_.copy, "message doesn't support `copy`", std::invalid_argument
        );
        return callbacks_.copy(*this, reservation);
    }

    /**
     * @brief Move this message into the memory of a reservation, leaving it empty.
     *
     * Invokes the registered `move` callback, which consumes this message, so its
     * memory is released rather than duplicated. Falls back to the `copy` callback when
     * no `move` callback is registered. Either way this message is reset afterwards, like
     * `release()`.
     *
     * If the `move` callback throws, this message is still reset, so its payload is lost.
     *
     * @param reservation Memory reservation to consume for the move.
     * @return A new `Message` instance holding the payload in the reserved memory.
     *
     * @throws std::invalid_argument if the message supports neither moving nor copying.
     */
    [[nodiscard]] Message move(MemoryReservation& reservation) {
        Message ret;
        if (callbacks_.move) {
            // The callback consumes `*this` so we make a copy.
            auto cb = callbacks_.move;
            try {
                ret = cb(std::move(*this), reservation);
            } catch (...) {
                reset();
                throw;
            }
        } else {
            RAPIDSMPF_EXPECTS(
                callbacks_.copy, "message doesn't support `move`", std::invalid_argument
            );
            ret = callbacks_.copy(*this, reservation);
        }
        reset();
        return ret;
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
    std::uint64_t sequence_number_{0};
    std::any payload_;
    ContentDescription content_description_;
    Callbacks callbacks_;
};

}  // namespace rapidsmpf::streaming
