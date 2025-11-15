/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <rapidsmpf/buffer/content_description.hpp>
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
    /// @brief Unique identifier assigned to each message.
    using MessageId = std::uint64_t;

    SpillableMessages() = default;
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

        Item() noexcept = default;

        Item(Message&& message) : message(std::move(message)) {}
    };

    // Never lock the global mutex and an item's mutex at the same time!
    mutable std::mutex global_mutex_;
    MessageId counter_{0};
    std::unordered_map<MessageId, std::shared_ptr<Item>> items_;
    mutable std::map<MessageId, ContentDescription> content_descriptions_;
};

}  // namespace rapidsmpf::streaming
