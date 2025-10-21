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
#include <rapidsmpf/communicator/message.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::communicator {

/**
 * @brief Type-agnostic communication interface.
 *
 * This interface provides a high-level, stateful communication management layer that
 * works with any message type implementing Message. It encapsulates the entire
 * communication protocol and state machine, providing coarse-grained operations that
 * can drive communication forward while maintaining full control over the underlying
 * communication patterns and optimizations.
 */
class CommunicationInterface {
  public:
    virtual ~CommunicationInterface() = default;

    /**
     * @brief Submit outgoing messages for communication.
     *
     * Takes ownership of ready messages and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * @param messages Vector of messages ready to be sent to remote ranks.
     */
    virtual void submit_outgoing_messages(
        std::vector<std::unique_ptr<Message>>&& messages
    ) = 0;

    /**
     * @brief Process all pending communication operations.
     *
     * Advances the communication state machine by:
     * - Receiving incoming message metadata
     * - Setting up data transfers
     * - Handling completed data transfers
     * - Cleaning up completed operations
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     * @return Vector of completed messages ready for local processing.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Message>> process_communication(
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
 * @brief Tag-based implementation of CommunicationInterface.
 *
 * This implementation provides the same communication protocol as
 * TagCommunicationInterface but works with the abstract Message instead of
 * concrete Chunk types.
 */
class TagCommunicationInterface : public CommunicationInterface {
  public:
    /**
     * @brief Constructor for TagCommunicationInterface.
     *
     * @param comm The communicator to use for operations.
     * @param op_id The operation ID for tagging messages.
     * @param rank The current rank (for logging and validation).
     * @param statistics The statistics to use for tracking communication operations.
     */
    TagCommunicationInterface(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        Rank rank,
        std::shared_ptr<Statistics> statistics
    );

    /**
     * @copydoc CommunicationInterface::submit_outgoing_messages
     *
     * @throw std::runtime_error if a message is sent to itself or if an outgoing
     * message already exists.
     */
    void submit_outgoing_messages(
        std::vector<std::unique_ptr<Message>>&& messages
    ) override;

    /**
     * @copydoc CommunicationInterface::process_communication
     */
    std::vector<std::unique_ptr<Message>> process_communication(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    ) override;

    /**
     * @copydoc CommunicationInterface::is_idle
     */
    bool is_idle() const override;

  private:
    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank rank_;
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
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
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
