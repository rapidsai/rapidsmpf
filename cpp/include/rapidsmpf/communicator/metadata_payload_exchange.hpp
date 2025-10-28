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
        [[nodiscard]] Rank peer_rank() const;

        /**
         * @brief Get the serialized metadata for this message.
         *
         * This metadata is sent first to inform the receiver about the incoming message.
         *
         * @return The serialized metadata.
         */
        [[nodiscard]] std::vector<std::uint8_t> const& metadata() const;

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

      private:
        friend class TagMetadataPayloadExchange;

        /**
         * @brief Set the data buffer for this message.
         *
         * This method is used internally by the communication interface.
         *
         * @param buffer Data buffer to be set.
         */
        void set_data(std::unique_ptr<Buffer> buffer);

        /**
         * @brief Get the unique message ID.
         *
         * This ID is used to track the message through the communication protocol and
         * must be unique within the context of a single communication operation.
         *
         * @return The unique message ID.
         */
        [[nodiscard]] std::uint64_t message_id() const;

        /**
         * @brief Set the message ID.
         *
         * This is used internally by the communication interface to assign
         * sequential message IDs.
         *
         * @param id Message ID to assign.
         */
        void set_message_id(std::uint64_t id);

        /**
         * @brief Get the expected payload size.
         *
         * This method is used internally by the communication interface.
         *
         * @return The expected size of the payload data in bytes.
         */
        [[nodiscard]] std::size_t expected_payload_size() const;

        /**
         * @brief Set the expected payload size.
         *
         * This is used internally by the communication interface to store the
         * payload size extracted from the metadata.
         *
         * @param size Expected payload size in bytes.
         */
        void set_expected_payload_size(std::size_t size);

        Rank peer_rank_;
        std::vector<std::uint8_t> metadata_;
        std::unique_ptr<Buffer> data_;
        std::uint64_t message_id_{0};
        std::size_t expected_payload_size_{0};
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
    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Tag metadata_tag_;
    Tag gpu_data_tag_;

    // Communication state containers
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::multimap<Rank, std::unique_ptr<Message>>
        incoming_messages_;  ///< Messages ready to be received.
    std::unordered_map<std::uint64_t, std::unique_ptr<Message>>
        in_transit_messages_;  ///< Messages currently in transit.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit messages.

    // Statistics tracking
    std::shared_ptr<Statistics> statistics_;

    // Sequential message ID generator
    std::uint64_t next_message_id_{0};

    /**
     * @brief Receive metadata for incoming messages.
     */
    void receive_metadata();

    /**
     * @brief Setup data receives for incoming messages.
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     *
     * @throw std::runtime_error if an in-transit message or future is not found, or
     * if a data buffer is not available.
     */
    void setup_data_receives(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    );

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
