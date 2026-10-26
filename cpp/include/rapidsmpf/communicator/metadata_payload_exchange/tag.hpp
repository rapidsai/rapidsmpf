/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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
     * See `MetadataPayloadExchange` for how receive buffers are allocated, how the
     * allocation timeout forces progress, and when payloads are moved to disk.
     *
     * @param comm The communicator to use for operations.
     * @param op_id The operation ID for tagging messages.
     * @param peers The ranks to communicate with. See `MetadataPayloadExchange`.
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     * @param allocation_timeout Progress timeout for receive buffer allocations.
     * @param br Buffer resource used to move payloads to disk and whose statistics track
     * communication operations. Must not be `nullptr`.
     *
     * @throws std::out_of_range If a peer is not a rank of @p comm.
     * @throws std::invalid_argument If @p peers contains the calling rank or duplicates.
     */
    TagMetadataPayloadExchange(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        std::vector<Rank> peers,
        AllocateBufferFn allocate_buffer_fn,
        Duration allocation_timeout,
        BufferResource* br
    );

    /**
     * @copydoc MetadataPayloadExchange::send
     *
     * @throw std::invalid_argument if the destination is not a peer or the message's
     * payload is disk-backed.
     */
    void send(std::unique_ptr<Message> message) override;

    // clang-format off
    /**
     * @copydoc MetadataPayloadExchange::send(std::vector<std::unique_ptr<Message>>&& messages);
     *
     * @throw std::invalid_argument if a destination is not a peer or a message's
     * payload is disk-backed.
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
     * Receive buffers are allocated in message order per peer. While a message from a
     * peer waits for a buffer, later messages from that peer wait behind it.
     *
     * Any exception thrown by the receive buffer allocator, for example when it is not
     * allowed to force an allocation, propagates out of this function and is a fatal,
     * unrecoverable error.
     *
     * When a forced receive buffer is moved to disk, this call blocks until the payload
     * is written.
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
        std::uint64_t arrival{0};  ///< Local arrival order, used to pick the oldest.
        /// Whether the receive buffer was allocated by forcing progress.
        bool forced_allocation{false};

        TagMessage(
            std::unique_ptr<Message> msg,
            std::uint64_t id = 0,
            std::size_t size = 0,
            std::uint64_t arrival_order = 0
        )
            : message(std::move(msg)),
              message_id(id),
              expected_payload_size(size),
              arrival(arrival_order) {}

        /**
         * @brief Whether the message is waiting for a receive buffer.
         *
         * @return True if the message has a payload but no buffer yet.
         */
        [[nodiscard]] bool is_awaiting_buffer() const {
            return expected_payload_size > 0 && message->data() == nullptr;
        }
    };

    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank const nranks_;
    Rank const rank_;
    // Peers in increasing order.
    std::vector<Rank> const peers_;
    // Whether a rank is a peer, indexed by Rank.
    std::vector<bool> is_peer_;
    Tag const metadata_tag_;
    Tag const gpu_data_tag_;
    AllocateBufferFn allocate_buffer_fn_;
    Duration const allocation_timeout_;
    BufferResource* br_;

    // Receive buffer allocation state:
    // - `num_awaiting_buffer_`: messages waiting for a buffer, per peer (by Rank).
    // - `total_awaiting_buffer_`: sum of `num_awaiting_buffer_`.
    // - `allocation_stall_start_`: start of the current period without a successful
    //   receive buffer allocation, set while at least one message is waiting.
    // - `next_arrival_`: arrival counter for incoming messages.
    std::vector<std::size_t> num_awaiting_buffer_;
    std::size_t total_awaiting_buffer_{0};
    std::optional<Clock::time_point> allocation_stall_start_{};
    std::uint64_t next_arrival_{0};

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
     * @throw std::runtime_error if an in-transit message or future is not found.
     */
    std::vector<std::unique_ptr<Message>> setup_data_receives();

    /**
     * @brief Try to allocate a receive buffer and attach it to a message waiting for one.
     *
     * On success, the new (empty) buffer is set on @p tag_msg, to be filled when the
     * data receive completes, and the waiting counts and stall timer are updated.
     *
     * @param tag_msg The message, which must be waiting for a buffer.
     * @param mode Passed on to the allocator. With `AllocationMode::MUST_SUCCEED`, a
     * buffer is always attached, so the function never returns false.
     * @return True if a buffer was attached.
     *
     * @throw std::runtime_error if the allocator returns `nullptr` although
     * the allocation must succeed.
     */
    bool try_attach_receive_buffer(TagMessage& tag_msg, AllocationMode mode);

    /**
     * @brief Force progress if receive buffer allocation has stalled for too long.
     *
     * When no receive buffer could be allocated within the allocation timeout, allocate
     * one for the oldest waiting message with `AllocationMode::MUST_SUCCEED`. Any
     * exception thrown by the allocator propagates.
     */
    void force_allocation_progress_if_stalled();

    /**
     * @brief Move a received payload to disk, if possible.
     *
     * The payload is moved only if the buffer resource has a disk resource and disk
     * space can be reserved. The move blocks until the payload is written.
     *
     * @param buffer The received payload, replaced by the on-disk payload if it was
     * moved.
     */
    void try_move_to_disk(std::unique_ptr<Buffer>& buffer);

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
