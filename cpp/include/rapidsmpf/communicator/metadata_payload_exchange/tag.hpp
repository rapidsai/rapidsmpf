/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

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
     * retried by subsequent calls to `progress()` until `allocation_timeout` expires.
     * @param statistics The statistics to use for tracking communication operations.
     * @param allocation_timeout How long receive-buffer allocation for a message may
     * keep failing before `progress()` throws.
     */
    TagMetadataPayloadExchange(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
        std::shared_ptr<Statistics> statistics,
        Duration allocation_timeout = std::chrono::seconds{30}
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
     * @throws std::runtime_error if receive-buffer allocation times out.
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
        std::unique_ptr<Message> message;
        std::uint64_t message_id{0};
        std::size_t expected_payload_size{0};
        /// Time of the first failed buffer allocation, used to time out retries.
        std::optional<Clock::time_point> first_allocation_failure;

        /**
         * @brief Construct a TagMessage.
         *
         * @param msg The wrapped message.
         * @param id The message ID.
         * @param size The expected payload size in bytes.
         */
        TagMessage(
            std::unique_ptr<Message> msg, std::uint64_t id = 0, std::size_t size = 0
        );

        /**
         * @brief Set the receive buffer of the message.
         *
         * The message's data buffer must not already be set. A `nullptr` buffer
         * marks a failed allocation attempt. The first failed attempt starts the
         * allocation timer; subsequent failed attempts throw once `timeout` has
         * elapsed.
         *
         * @param buffer The allocated buffer, or `nullptr` if allocation failed.
         * @param timeout How long allocation may keep failing before throwing.
         * @return `true` if the buffer was set, `false` if allocation failed.
         *
         * @throw std::logic_error if the message's data buffer is already set.
         * @throw std::runtime_error if `buffer` is `nullptr` and the allocation
         * timeout has elapsed.
         */
        bool set_buffer(std::unique_ptr<Buffer> buffer, Duration timeout);

        /**
         * @brief Whether the message's data buffer is set.
         *
         * A message without a payload never needs a buffer, so it is considered set.
         *
         * @return True if the message has no payload or its data buffer is set,
         * false otherwise.
         */
        [[nodiscard]] bool is_buffer_set() const {
            return expected_payload_size == 0 || message->data() != nullptr;
        }
    };

    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank const nranks_;
    Rank const rank_;
    Tag const metadata_tag_;
    Tag const gpu_data_tag_;
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn_;
    Duration const allocation_timeout_;

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
     * receive-buffer allocation times out.
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

    /**
     * @brief Whether payload allocation is deferred for a peer.
     *
     * Payload buffers are allocated in message order, so allocation is deferred
     * while any incoming message from the peer is still waiting for a buffer.
     *
     * @param peer The source rank.
     * @return True if new messages from `peer` must not allocate a payload buffer.
     */
    [[nodiscard]] bool is_allocation_deferred(Rank peer) const;
};


}  // namespace rapidsmpf::communicator
