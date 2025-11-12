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
 *
 * @note All concrete implementations are expected to provide a constructor with
 * the following signature:
 * @code
 * DerivedMetadataPayloadExchange(
 *     std::shared_ptr<Communicator> comm,
 *     OpID op_id,
 *     std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
 *     std::shared_ptr<Statistics> statistics
 * );
 * @endcode
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
            std::vector<std::uint8_t>&& metadata,
            std::unique_ptr<Buffer> data = nullptr
        );

        /**
         * @brief Get the destination rank for outgoing or source rank for incoming
         * messages.
         *
         * @return The rank of the destination or source.
         */
        [[nodiscard]] constexpr Rank peer_rank() const noexcept {
            return peer_rank_;
        }

        /**
         * @brief Get the serialized metadata for this message.
         *
         * This metadata is sent first to inform the receiver about the incoming message.
         *
         * @return The serialized metadata.
         */
        [[nodiscard]] constexpr std::vector<std::uint8_t> const&
        metadata() const noexcept {
            return metadata_;
        }

        /**
         * @brief Release ownership of the metadata.
         *
         * This is typically called when transferring metadata to the communication layer.
         *
         * @return Metadata with ownership transferred.
         */
        [[nodiscard]] std::vector<std::uint8_t> release_metadata() noexcept;

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
        [[nodiscard]] std::unique_ptr<Buffer> release_data() noexcept;

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
     * @brief Send a single message to a remote rank.
     *
     * Takes ownership of a ready message and manages its transmission, including
     * metadata sending and coordination of data transfer.
     *
     * The messages sent from the calling process to a destination remote rank are
     * guaranteed to be received in the same order as they were sent. No ordering is
     * guaranteed between messages sent to different remote ranks.
     *
     * @param message Message ready to be sent to a remote rank.
     */
    virtual void send(std::unique_ptr<Message> message) = 0;

    /**
     * @brief Send messages to remote ranks.
     *
     * Takes ownership of ready messages and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * The messages sent from the calling process to a destination remote rank are
     * guaranteed to be received in the same order as they were sent. No ordering is
     * guaranteed between messages sent to different remote ranks.
     *
     * @param messages Vector of messages ready to be sent to remote ranks.
     */
    virtual void send(std::vector<std::unique_ptr<Message>>&& messages) = 0;

    /**
     * @brief Progress the communication state machine.
     *
     * Advances the internal state of the communication layer by processing pending
     * operations such as receiving metadata, setting up data transfers, completing
     * data transfers, and cleaning up completed operations. Completed messages are
     * stored internally and can be retrieved via recv().
     *
     * This method should be called periodically to make progress on communication.
     */
    virtual void progress() = 0;

    /**
     * @brief Receive messages from remote ranks.
     *
     * The messages received by the calling process are guaranteed to be received in the
     * same order as they were sent by the source remote rank. No ordering is guaranteed
     * between messages received from different remote ranks.
     *
     * @return Vector of completed messages ready for local processing.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Message>> recv() = 0;

    /**
     * @brief Check if the communication layer is currently idle.
     *
     * Indicates whether there are any active or pending communication operations.
     * A return value of `true` means the exchange is idling, i.e. no operations
     * are currently in progress. However, new send/receive requests may still be
     * submitted in the future; this does not imply that all communication has been
     * fully finalized or globally synchronized.
     *
     * @return `true` if the communication layer is idle; `false` if activity is ongoing.
     */
    [[nodiscard]] virtual bool is_idle() const = 0;
};


}  // namespace rapidsmpf::communicator
