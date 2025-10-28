/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::communicator {

/**
 * @brief Interface for exchanging serialized metadata and payload between ranks.
 *
 * The `MetadataPayloadExchange` class defines an abstract interface for transmitting
 * messages that contain both serialized metadata and a data payload. This abstraction
 * simplifies scenarios where metadata and payload must be exchanged together as a
 * single logical unit.
 *
 * Concrete implementations, such as `TagMetadataPayloadExchange`, use the
 * `Communicator` to implement this interface. In the future, other implementations
 * may leverage specialized features beyond the basic `Communicator` API to further
 * optimize this communication pattern.
 *
 * @note This class is not thread-safe. All methods must be called from the same thread.
 */
class MetadataPayloadExchange {
  public:
    /**
     * @brief Message class for communication.
     *
     * This class contains the essential information needed for communication:
     * data payload, metadata, and peer rank (source/destination).
     */
    class Message {
      public:
        /**
         * @brief Construct a new Message.
         *
         * @param peer_rank Destination (outgoing) or source (incoming) rank.
         * @param metadata Serialized metadata.
         * @param data Data buffer (can be nullptr for metadata-only messages).
         */
        Message(
            Rank peer_rank,
            std::vector<std::uint8_t> metadata,
            std::unique_ptr<Buffer> data = nullptr
        );

        /**
         * @brief Get the destination rank for outgoing or source rank for incoming
         * messages.
         *
         * @return The rank of the destination or source.
         */
        [[nodiscard]] constexpr Rank peer_rank() const {
            return peer_rank_;
        }

        /**
         * @brief Get the serialized metadata for this message.
         *
         * This metadata is sent first to inform the receiver about the incoming message.
         *
         * @return The serialized metadata.
         */
        [[nodiscard]] constexpr std::vector<std::uint8_t> const& metadata() const {
            return metadata_;
        }

        /**
         * @brief Get the data buffer for this message.
         *
         * @return The data buffer, or nullptr if no data.
         */
        [[nodiscard]] Buffer const* data() const;

        /**
         * @brief Release ownership of the data buffer.
         *
         * This is typically called when transferring a buffer to the communication layer.
         *
         * @return Data buffer with ownership transferred, or nullptr if no data.
         */
        [[nodiscard]] std::unique_ptr<Buffer> release_data();

        /**
         * @brief Set the data buffer for this message.
         *
         * This method can be used by implementations to update the data buffer.
         *
         * @param buffer Data buffer to be set.
         */
        void set_data(std::unique_ptr<Buffer> buffer);

      private:
        Rank peer_rank_;
        std::vector<std::uint8_t> metadata_;
        std::unique_ptr<Buffer> data_;
    };

    virtual ~MetadataPayloadExchange() = default;

    /**
     * @brief Send messages to remote ranks.
     *
     * Takes ownership of ready messages and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * @param messages Vector of messages ready to be sent to remote ranks.
     */
    virtual void send_messages(std::vector<std::unique_ptr<Message>>&& messages) = 0;

    /**
     * @brief Send a single message to a remote rank.
     *
     * Takes ownership of a ready message and manages its transmission, including
     * metadata sending and coordination of data transfer.
     *
     * @param message Message ready to be sent to a remote rank.
     */
    virtual void send_message(std::unique_ptr<Message> message) = 0;

    /**
     * @brief Receive messages from remote ranks.

     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     * @return Vector of completed messages ready for local processing.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Message>> receive_messages(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    ) = 0;

    /**
     * @brief Check if all communication operations are complete.
     *
     * @return True if no pending operations remain, false otherwise.
     */
    [[nodiscard]] virtual bool is_idle() const = 0;
};

/**
 * @brief Tag-based implementation of MetadataPayloadExchange.
 *
 * This implementation provides the same communication protocol as
 * TagMetadataPayloadExchange but works with the abstract Message.
 */
class TagMetadataPayloadExchange : public MetadataPayloadExchange {
  public:
    /**
     * @brief Constructor for TagMetadataPayloadExchange.
     *
     * @param comm The communicator to use for operations.
     * @param op_id The operation ID for tagging messages.
     * @param statistics The statistics to use for tracking communication operations.
     */
    TagMetadataPayloadExchange(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        std::shared_ptr<Statistics> statistics
    );

    /**
     * @copydoc MetadataPayloadExchange::send_messages
     *
     * @throw std::runtime_error if a message is sent to itself or if an outgoing
     * message already exists.
     */
    void send_messages(std::vector<std::unique_ptr<Message>>&& messages) override;

    /**
     * @copydoc MetadataPayloadExchange::send_message
     *
     * @throw std::runtime_error if a message is sent to itself or if an outgoing
     * message already exists.
     */
    void send_message(std::unique_ptr<Message> message) override;

    /**
     * @copydoc MetadataPayloadExchange::receive_messages
     *
     * Advances the communication state machine by:
     * - Receiving incoming message metadata
     * - Setting up data transfers
     * - Handling completed data transfers
     * - Cleaning up completed operations
     */
    std::vector<std::unique_ptr<Message>> receive_messages(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    ) override;

    /**
     * @copydoc MetadataPayloadExchange::is_idle
     */
    bool is_idle() const override;

  private:
    /**
     * @brief Internal wrapper for tracking protocol-specific state.
     *
     * This struct wraps a Message with protocol-specific tracking information
     * that is used internally by TagMetadataPayloadExchange but not exposed
     * through the public interface.
     */
    struct TagMessage {
        std::unique_ptr<Message> message;
        std::uint64_t message_id{0};
        std::size_t expected_payload_size{0};

        TagMessage(
            std::unique_ptr<Message> msg, std::uint64_t id = 0, std::size_t size = 0
        )
            : message(std::move(msg)), message_id(id), expected_payload_size(size) {}
    };

    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Tag const metadata_tag_;
    Tag const gpu_data_tag_;

    // Communication state containers
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::unordered_map<Rank, std::vector<TagMessage>>
        incoming_messages_;  ///< Messages ready to be received, grouped by rank.
    std::unordered_map<std::uint64_t, TagMessage>
        in_transit_messages_;  ///< Messages currently in transit.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit messages.

    // Statistics tracking
    std::shared_ptr<Statistics> statistics_;

    // Sequential message ID generator
    std::uint64_t next_message_id_{0};

    /**
     * @brief Receive metadata for incoming messages.
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     */
    void receive_metadata(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    );

    /**
     * @brief Setup data receives for incoming messages.
     *
     * @return A vector of completed metadata-only messages.
     *
     * @throw std::runtime_error if an in-transit message or future is not found, or
     * if a data buffer is not available.
     */
    std::vector<std::unique_ptr<Message>> setup_data_receives();

    /**
     * @brief Complete data transfers for in-transit messages.
     *
     * @return A vector of completed messages.
     *
     * @throw std::runtime_error if an in-transit message or future is not found, or
     * if a data buffer is not available
     */
    std::vector<std::unique_ptr<Message>> complete_data_transfers();

    /**
     * @brief Cleanup completed operations (fire-and-forget sends and receives).
     */
    void cleanup_completed_operations();
};


}  // namespace rapidsmpf::communicator
