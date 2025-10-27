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

class TagMetadataPayloadExchange;

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
     * @brief Get the destination rank for outgoing or source rank for incoming messages.
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
     * This ID is used to track the message through the communication protocol and must be
     * unique within the context of a single communication operation.
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

}  // namespace rapidsmpf::communicator
