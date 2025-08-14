/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Self-contained communication manager for shuffler operations.
 *
 * This interface provides a high-level, stateful communication management layer for the
 * shuffler. Unlike a simple wrapper, it owns and manages all communication-related state,
 * including pending operations, incoming/outgoing chunks, and coordination logic.
 *
 * The interface encapsulates the entire communication protocol and state machine,
 * providing coarse-grained operations that the shuffler's progress loop can call
 * to drive the communication forward while maintaining full control over the
 * underlying communication patterns and optimizations.
 */
class CommunicationInterface {
  public:
    virtual ~CommunicationInterface() = default;

    /**
     * @brief Submit outgoing chunks for communication.
     *
     * Takes ownership of ready chunks and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * @param chunks Vector of chunks ready to be sent to remote ranks.
     * @param partition_owner Function to determine destination rank for each chunk.
     * @param br Buffer resource for communication operations.
     */
    virtual void submit_outgoing_chunks(
        std::vector<detail::Chunk>&& chunks,
        std::function<Rank(PartID)> partition_owner,
        BufferResource* br
    ) = 0;

    /**
     * @brief Process all pending communication operations.
     *
     * Advances the communication state machine by:
     * - Receiving incoming chunk metadata and setting up data transfers
     * - Processing ready-for-data acknowledgments and sending data
     * - Completing data transfers and making chunks available
     * - Cleaning up completed operations
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     * @param stream CUDA stream for memory operations.
     * @param br Buffer resource for communication operations.
     * @return Vector of completed chunks ready for local processing.
     */
    [[nodiscard]] virtual std::vector<detail::Chunk> process_communication(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) = 0;

    /**
     * @brief Check if all communication operations are complete.
     *
     * @return True if no pending operations remain, false otherwise.
     */
    [[nodiscard]] virtual bool is_idle() const = 0;
};

/**
 * @brief Tag implementation of CommunicationInterface.
 *
 * This implementation owns and manages all communication state that was previously
 * held in the Progress class. It replicates the exact current communication behavior
 * while providing a self-contained, stateful communication manager.
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

    void submit_outgoing_chunks(
        std::vector<detail::Chunk>&& chunks,
        std::function<Rank(PartID)> partition_owner,
        BufferResource* br
    ) override;

    std::vector<detail::Chunk> process_communication(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override;

    bool is_idle() const override;

  private:
    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank rank_;
    Tag ready_for_data_tag_;
    Tag metadata_tag_;
    Tag gpu_data_tag_;

    // Communication state containers (moved from Progress class)
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::multimap<Rank, detail::Chunk>
        incoming_chunks_;  ///< Chunks ready to be received.
    std::unordered_map<detail::ChunkID, detail::Chunk>
        outgoing_chunks_;  ///< Chunks ready to be sent.
    std::unordered_map<detail::ChunkID, detail::Chunk>
        in_transit_chunks_;  ///< Chunks currently in transit.
    std::unordered_map<detail::ChunkID, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit chunks.
    std::unordered_map<Rank, std::vector<std::unique_ptr<Communicator::Future>>>
        ready_ack_receives_;  ///< Receives matching ready for data messages.

    // Statistics tracking
    std::shared_ptr<Statistics> statistics_;

    // Helper methods for the communication protocol phases
    void send_metadata_phase(
        std::function<Rank(PartID)> partition_owner, BufferResource* br
    );
    void receive_metadata_phase();
    void setup_data_receives_phase(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
        rmm::cuda_stream_view stream,
        BufferResource* br
    );
    void process_ready_acks_phase();
    std::vector<detail::Chunk> complete_data_transfers_phase();
    void cleanup_completed_operations();
};

}  // namespace rapidsmpf::shuffler
