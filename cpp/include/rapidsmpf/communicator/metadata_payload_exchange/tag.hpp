/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::communicator {

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
     * @param allocate_buffer_fn Function to allocate buffers for incoming data. May
     * return `nullptr` when allocation is temporarily unavailable; allocation is
     * retried by subsequent calls to `progress()` up to the allocation retry limit.
     * @param statistics The statistics to use for tracking communication operations.
     */
    TagMetadataPayloadExchange(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
        std::shared_ptr<Statistics> statistics
    );

    /**
     * @copydoc MetadataPayloadExchange::send
     *
     * @throw std::runtime_error if a message is sent to itself or if an outgoing
     * message already exists.
     */
    void send(std::unique_ptr<Message> message) override;

    // clang-format off
    /**
     * @copydoc MetadataPayloadExchange::send(std::vector<std::unique_ptr<Message>>&& messages);
     *
     * @throw std::runtime_error if a message is sent to itself or if an outgoing
     * message already exists.
     */
    // clang-format on
    void send(std::vector<std::unique_ptr<Message>>&& messages) override;

    /**
     * @copydoc MetadataPayloadExchange::progress
     *
     * Advances the communication state machine by:
     * - Receiving incoming message metadata
     * - Setting up data transfers
     * - Handling completed data transfers
     * - Cleaning up completed operations
     *
     * @throws std::runtime_error if receive-buffer allocation retries are exhausted.
     */
    void progress() override;

    /**
     * @copydoc MetadataPayloadExchange::recv
     */
    std::vector<std::unique_ptr<Message>> recv() override;

    /**
     * @copydoc MetadataPayloadExchange::finish
     */
    void finish() override;

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
        /// Default number of progress iterations a receive-buffer allocation may
        /// fail before the exchange gives up. Overridable per exchange, see
        /// `allocation_retry_limit_`.
        static constexpr std::int32_t default_allocation_retry_limit{8};

        std::unique_ptr<Message> message;
        std::uint64_t message_id{0};
        std::size_t expected_payload_size{0};
        std::int32_t allocation_retries_remaining{default_allocation_retry_limit};

        TagMessage(
            std::unique_ptr<Message> msg,
            std::uint64_t id = 0,
            std::size_t size = 0,
            std::int32_t retry_limit = default_allocation_retry_limit
        )
            : message(std::move(msg)),
              message_id(id),
              expected_payload_size(size),
              allocation_retries_remaining(retry_limit) {
            if (expected_payload_size > 0 && message->data() == nullptr) {
                --allocation_retries_remaining;
            }
        }
    };


    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank const nranks_;
    Rank const rank_;
    Tag const metadata_tag_;
    Tag const gpu_data_tag_;
    /**
     * @brief Progress iterations a receive-buffer allocation may fail before the
     * exchange throws ("receive buffer allocation retries exhausted").
     *
     * Read from the environment variable `RAPIDSMPF_TAG_ALLOCATION_RETRY_LIMIT` at
     * construction, default `TagMessage::default_allocation_retry_limit`. The default
     * (a few milliseconds of progress iterations) assumes the allocation callback can
     * always make room by spilling; when the buffer resource is shared with other
     * consumers that hold device memory for longer (e.g. a streaming pipeline), the
     * limit must be far higher so the receive waits for them instead of aborting.
     */
    std::int32_t allocation_retry_limit_;
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn_;

    // Per-peer tracking for op_id reuse (see rapidsai/rapidsmpf#927).
    // After finish() is called, termination markers are exchanged so each peer
    // knows exactly how many application messages to expect. recv_from is used
    // instead of recv_any to avoid consuming messages from a future collective.
    bool finished_{false};
    std::vector<std::size_t>
        messages_sent_to_;  ///< Application messages sent per peer, indexed by Rank.
    std::vector<std::size_t>
        peer_received_;  ///< Application messages received per peer, indexed by Rank.
    std::vector<std::size_t>
        peer_expected_;  ///< Expected application messages per peer (0 = unknown).
    std::vector<bool>
        peer_terminated_;  ///< Whether we received the termination marker from each peer.
    std::vector<bool>
        peer_allocation_deferred_;  ///< Whether payload allocation is deferred per peer.

    /// Sentinel message_id value used to identify protocol-level termination markers.
    static constexpr std::uint64_t termination_sentinel_ = UINT64_MAX;

    // Communication state containers
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::unordered_map<Rank, std::vector<TagMessage>>
        incoming_messages_;  ///< Messages ready to be received, grouped by rank.
    std::unordered_map<Rank, std::vector<TagMessage>>
        in_transit_messages_;  ///< Messages currently in transit, grouped by rank in
                               ///< order.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit messages.
    std::vector<std::unique_ptr<Message>>
        received_messages_;  ///< Messages that have completed and are ready to be
                             ///< retrieved.

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
     * @return A vector of completed metadata-only messages.
     *
     * @throw std::runtime_error if an in-transit message or future is not found, or
     * receive-buffer allocation retries are exhausted.
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
