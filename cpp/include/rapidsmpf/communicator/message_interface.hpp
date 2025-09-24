/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf::communicator {

/**
 * @brief Abstract interface for messages that can be sent/received through
 * CommunicationInterface.
 *
 * This interface decouples the communication layer from specific message types like
 * Chunk, allowing the CommunicationInterface to work with any message type that
 * implements this interface. Each message contains a destination/source rank, one or more
 * data buffers, and a unique ID.
 */
class MessageInterface {
  public:
    virtual ~MessageInterface() = default;

    /**
     * @brief Get the unique message ID.
     *
     * This ID is used to track the message through the communication protocol and must be
     * unique within the context of a single communication operation.
     *
     * @return The unique message ID.
     */
    [[nodiscard]] virtual std::uint64_t message_id() const = 0;

    /**
     * @brief Get the destination rank for outgoing or source rank for incoming messages.
     *
     * @return The rank of the destination or source.
     */
    [[nodiscard]] virtual Rank peer_rank() const = 0;

    /**
     * @brief Get the serialized metadata for this message.
     *
     * This metadata is sent first to inform the receiver about the incoming message.
     * The format is implementation-specific but should contain enough information
     * for the receiver to prepare for data reception.
     *
     * @return The serialized metadata.
     */
    [[nodiscard]] virtual std::vector<std::uint8_t> serialize_metadata() const = 0;

    /**
     * @brief Get the total size of all data buffers.
     *
     * @return Total size in bytes of all data buffers.
     */
    [[nodiscard]] virtual std::size_t total_data_size() const = 0;

    /**
     * @brief Check if data buffers are ready for communication.
     *
     * @return True if all required data buffers are ready.
     */
    [[nodiscard]] virtual bool is_data_ready() const = 0;

    /**
     * @brief Set the data buffer for this message.
     *
     * @param buffer Data buffer to be sent/received.
     */
    virtual void set_data_buffer(std::unique_ptr<Buffer> buffer) = 0;

    /**
     * @brief Release ownership of the data buffer.
     *
     * This is typically called when transferring a buffer to the communication layer.
     *
     * @return Data buffer with ownership transferred.
     */
    [[nodiscard]] virtual std::unique_ptr<Buffer> release_data_buffer() = 0;

    /**
     * @brief Get the memory type of the data buffers.
     *
     * @return The memory type (device, host, etc.) of the data buffers.
     */
    [[nodiscard]] virtual MemoryType data_memory_type() const = 0;

    /**
     * @brief Check if the message is ready for processing.
     *
     * This typically means all required buffers are set and any async operations
     * have completed.
     *
     * @return True if the message is ready for processing.
     */
    [[nodiscard]] virtual bool is_ready() const = 0;

    /**
     * @brief Create a string representation of the message for logging/debugging.
     *
     * @return String representation of the message.
     */
    [[nodiscard]] virtual std::string to_string() const = 0;
};

/**
 * @brief Factory interface for creating message instances.
 *
 * This allows the communication interface to create message instances without
 * knowing the specific implementation type.
 */
class MessageFactory {
  public:
    virtual ~MessageFactory() = default;

    /**
     * @brief Create a message instance from serialized metadata.
     *
     * @param metadata The serialized metadata.
     * @param peer_rank The peer rank (destination for outgoing or source for incoming
     * messages).
     * @return A new message instance.
     */
    [[nodiscard]] virtual std::unique_ptr<MessageInterface> create_from_metadata(
        std::vector<std::uint8_t> const& metadata, Rank peer_rank
    ) const = 0;

    /**
     * @brief Allocate a buffer for receiving message data.
     *
     * @param size Size of data to be received.
     * @param message The message instance for context.
     * @return Allocated buffer ready for receiving data.
     */
    [[nodiscard]] virtual std::unique_ptr<Buffer> allocate_receive_buffer(
        std::size_t size, MessageInterface const& message
    ) const = 0;
};

}  // namespace rapidsmpf::communicator
