/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Container for individually spillable messages.
 *
 * `SpillableMessages` manages a collection of `Message` instances that can be
 * spilled or extracted independently. Each message is assigned a unique
 * `MessageId` upon insertion, which can later be used to extract or spill that
 * message.
 *
 * The container is thread-safe for concurrent insertions, extractions, and
 * spills.
 *
 * ### Example
 * @code{.cpp}
 * SpillableMessages messages;
 * auto id = messages.insert(to_message(TableChunk{...}));
 * messages.spill(id, ...);
 * auto msg = messages.extract(id);
 * @endcode
 */
class SpillableMessages {
  public:
    /// @brief Order in which messages are chosen for spilling.
    enum class SpillOrder : std::uint8_t {
        EARLIEST_FIRST,  ///< Spill the earliest inserted messages first.
        LATEST_FIRST  ///< Spill the most recently inserted messages first.
    };

    /// @brief Unique identifier assigned to each message.
    using MessageId = std::uint64_t;

    /**
     * @brief Construct an empty container.
     *
     * @param spill_order Order in which `try_spill()` picks
     * messages. Defaults to earliest inserted first.
     * @param reservation_context Default telemetry context for messages inserted
     * without an explicit context.
     */
    explicit SpillableMessages(
        SpillOrder spill_order = SpillOrder::EARLIEST_FIRST,
        std::uint64_t reservation_context = 0
    )
        : spill_order_{spill_order}, default_reservation_context_{reservation_context} {}

    SpillableMessages(SpillableMessages const&) = delete;
    SpillableMessages& operator=(SpillableMessages const&) = delete;
    SpillableMessages(SpillableMessages&&) noexcept = delete;
    SpillableMessages& operator=(SpillableMessages&&) noexcept = delete;

    /**
     * @brief Insert a new message and return its assigned ID.
     *
     * @param message Message to insert.
     * @return Assigned `MessageId` of the inserted message.
     */
    [[nodiscard]] MessageId insert(Message&& message);

    /**
     * @brief Insert a new message with an explicit reservation context.
     *
     * @param message Message to insert.
     * @param reservation_context Telemetry context for future spill reservations.
     * @return Assigned `MessageId` of the inserted message.
     */
    [[nodiscard]] MessageId insert(Message&& message, std::uint64_t reservation_context);

    /**
     * @brief Extract and remove a message by ID.
     *
     * If the message is currently being spilled, this method blocks until spilling
     * completes.
     *
     * @param mid Message identifier.
     * @return Extracted `Message` instance.
     *
     * @throws std::out_of_range If the message ID is invalid or was already extracted.
     */
    [[nodiscard]] Message extract(MessageId mid);

    /**
     * @brief Create a deep copy of a message without removing it.
     *
     * This method duplicates the message identified by `mid` while leaving the
     * original message intact inside the container. The returned message is a
     * full deep copy of the payload. If the message is currently being spilled
     * by another thread, this call waits until spilling completes.
     *
     * @param mid Message identifier.
     * @param reservation Memory reservation used for allocating buffers during
     * the deep copy. The reservation also determines the memory type of the
     * returned message.
     *
     * @return A deep copy of the referenced `Message`.
     *
     * @throws std::out_of_range If the message has already been extracted or
     * the message identifier is invalid.
     * @throws std::runtime_error If required memory cannot be allocated using
     * the provided reservation.
     */
    [[nodiscard]] Message copy(MessageId mid, MemoryReservation& reservation);

    /**
     * @brief Spill a message's device memory to host memory.
     *
     * Performs an in-place deep copy of the message's payload from device to
     * host memory using the specified buffer resource.
     *
     * If the message is currently being accessed by another thread, is already
     * spilled, not spillable, or does not exist, the operation returns immediately
     * without spilling.
     *
     * @param mid Message identifier. If the message does not exist, zero is returned.
     * @param br Buffer resource used for allocations during the spill operation.
     * @return Number of bytes released from device memory (0 if nothing was spilled).
     *
     * @throws std::runtime_error If there is insufficient host memory to reserve.
     */
    [[nodiscard]] std::size_t spill(MessageId mid, BufferResource* br) const;

    /**
     * @brief Spill messages until the target amount of memory has been released.
     *
     * Messages are spilled in the order given by the `SpillOrder` passed at
     * construction (earliest inserted first by default). The function stops once
     * the cumulative amount of spilled memory reaches or exceeds `amount`, or when no
     * further spillable messages remain.
     *
     * @param br Buffer resource used for spill allocations.
     * @param amount Target number of bytes to spill.
     * @param mem_type The memory type to spill from. Only `MemoryType::DEVICE` is
     * supported for now, other types spill nothing.
     * @return The total number of bytes actually spilled.
     *
     * @todo Support additional spilling strategies (e.g., size-based or
     * priority-based).
     */
    [[nodiscard]] std::size_t try_spill(
        BufferResource* br, std::size_t amount, MemoryType mem_type
    ) const;

    /**
     * @brief Get a snapshot of current messages' content descriptions.
     *
     * The returned map may become outdated immediately if other threads
     * modify the container after this call.
     *
     * Use this snapshot to decide which messages to spill, but keep in mind
     * that the information may no longer be accurate when the actual spill
     * occurs. When calling `spill()`, the returned size reflects what was
     * actually spilled.
     *
     * @return Copy of a map from `MessageId` to `ContentDescription`.
     */
    std::map<MessageId, ContentDescription> get_content_descriptions() const;

    /**
     * @brief Get the content description of a message by ID.
     *
     * @param mid Message identifier.
     * @return Content description of the message.
     *
     * @throws std::out_of_range If the message does not exist.
     */
    ContentDescription get_content_description(MessageId mid) const;

    /**
     * @brief Clear all outstanding messages
     *
     * This is useful for avoiding Items from outliving the BufferResource on
     * which they were allocated. It is the caller's responsibility to clear
     * the messages before the BufferResource is destroyed.
     */
    void clear();

  private:
    /**
     * @brief Thread-safe item containing a `Message`.
     *
     * Each item is protected by its own mutex, enabling fine-grained exclusive access
     * to individual messages, as opposed to the `global_mutex_`, which guards insert
     * and extract operations.
     *
     * Because an item can be locked by one thread while another thread extracts it,
     * the contained `Message` is stored as an `std::optional`. If another thread has
     * already extracted the message, this can be detected by checking `has_value()`.
     *
     * An item's mutex and the global mutex must never be locked at the same time.
     */
    struct Item {
        mutable std::mutex mutex;
        std::optional<Message> message;
        std::uint64_t reservation_context;

        Item(Message&& message, std::uint64_t reservation_context)
            : message(std::move(message)), reservation_context(reservation_context) {}
    };

    // Never lock the global mutex and an item's mutex at the same time!
    mutable std::mutex global_mutex_;
    SpillOrder spill_order_;
    MessageId counter_{0};
    std::unordered_map<MessageId, std::shared_ptr<Item>> items_;
    mutable std::map<MessageId, ContentDescription> content_descriptions_;
    std::uint64_t default_reservation_context_;
};

}  // namespace rapidsmpf::streaming
