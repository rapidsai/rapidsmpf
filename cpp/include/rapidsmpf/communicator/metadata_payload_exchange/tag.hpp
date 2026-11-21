/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/rank_topology.hpp>
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
     * @brief What happens to the payload of a message whose receive buffer was forced.
     */
    enum class ForcedReceive : bool {
        MOVE_TO_DISK,  ///< The payload is moved to disk, if `br` has a disk resource.
        KEEP_IN_MEMORY,  ///< The payload stays in the forced receive buffer.
    };

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
     * @param forced_receive What happens to the payload of a message whose receive
     * buffer was forced.
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
        BufferResource* br,
        ForcedReceive forced_receive = ForcedReceive::MOVE_TO_DISK
    );

    /**
     * @brief Destructor.
     *
     * Logs a warning if the exchange is destroyed before `finish()` has been called and
     * `is_idle()` returns `true`.
     */
    ~TagMetadataPayloadExchange() noexcept override;

    /**
     * @copydoc MetadataPayloadExchange::send
     *
     * A DISK payload is handed over by path, without sending its bytes:
     * - With `AfterSend::DISCARD`, the file itself is handed over.
     * - With `AfterSend::RETAIN`, a clone of the file is handed over
     *   (`DiskResource::clone()`). On filesystems without reflink support, the clone
     *   copies the file and blocks this call while it does.
     *
     * @throw std::invalid_argument if the destination is not a peer, the message's
     * payload is not in `sendable_memory_types()` of its destination, or a DISK payload
     * does not belong to the disk resource of `br`.
     */
    void send(std::unique_ptr<Message> message) override;

    /**
     * @copydoc MetadataPayloadExchange::sendable_memory_types
     *
     * Returns `MEMORY_TYPES`, which includes `DISK`, if `br` had a `RankTopology`
     * attached when the exchange was constructed and @p dst can read this rank's spill
     * files (`RankTopology::can_read_spill_files()`). Otherwise returns
     * `ADDRESSABLE_MEMORY_TYPES`.
     *
     * A DISK payload is sent as the path of its file, and @p dst takes ownership of the
     * file. This assumes ranks are configured symmetrically, so a rank that can read
     * the file also has a disk resource to adopt it into.
     *
     * @throws std::out_of_range If a topology is attached and @p dst is not a valid
     * rank.
     */
    [[nodiscard]] std::span<MemoryType const> sendable_memory_types(
        Rank dst
    ) const override;

    // clang-format off
    /**
     * @copydoc MetadataPayloadExchange::send(std::vector<std::unique_ptr<Message>>&& messages);
     *
     * @throw std::invalid_argument if a destination is not a peer or a message's
     * payload is not in `sendable_memory_types()` of its destination.
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
     * @copydoc MetadataPayloadExchange::release_sent
     */
    std::vector<std::unique_ptr<Message>> release_sent() override;

    /**
     * @copydoc MetadataPayloadExchange::finish
     */
    void finish() override;

    /**
     * @copydoc MetadataPayloadExchange::is_finished
     */
    bool is_finished() const override;

    /**
     * @copydoc MetadataPayloadExchange::is_idle
     *
     * After finish(), this also waits until every peer that was handed DISK payloads by
     * path has confirmed that it took over the files. Once idle, the files this rank
     * handed over no longer depend on its spill directory, which may then be removed.
     */
    bool is_idle() const override;

  private:
    /// @brief A message's metadata and protocol trailer, defined in the source file.
    struct CombinedMetadata;

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
        /// File handed over by the sender for a DISK payload, empty otherwise.
        std::filesystem::path handoff_path{};

        TagMessage(
            std::unique_ptr<Message> msg,
            std::uint64_t id = 0,
            std::size_t size = 0,
            std::uint64_t arrival_order = 0,
            std::filesystem::path handoff = {}
        )
            : message(std::move(msg)),
              message_id(id),
              expected_payload_size(size),
              arrival(arrival_order),
              handoff_path(std::move(handoff)) {}

        /**
         * @brief Construct an incoming message from its deserialized metadata.
         *
         * @param combined The deserialized metadata and trailer, consumed.
         * @param peer The rank the message came from.
         * @param arrival_order The local arrival order.
         */
        TagMessage(CombinedMetadata&& combined, Rank peer, std::uint64_t arrival_order);

        /**
         * @brief Whether the message is waiting for a receive buffer.
         *
         * @return True if the message has a payload but no buffer yet.
         */
        [[nodiscard]] bool is_awaiting_buffer() const {
            return expected_payload_size > 0 && message->data() == nullptr;
        }

        /**
         * @brief Take ownership of the handed-over file as the message's DISK payload.
         *
         * Blocks while the file is copied if it is on a different filesystem than the
         * disk resource of @p br. `handoff_path` must be set, and is cleared.
         *
         * @param br The buffer resource whose disk resource adopts the file.
         * @param statistics Statistics to record the handed-over bytes in.
         *
         * @throw std::invalid_argument if the file does not exist, for example because
         * it is not visible to this rank.
         * @throw std::runtime_error if the file cannot be placed under the disk resource.
         */
        void adopt_handoff(BufferResource& br, Statistics& statistics);
    };

    /// @brief A message retained with `AfterSend::RETAIN` while its send is in flight.
    struct RetainedMessage {
        // The message, without its payload.
        std::unique_ptr<Message> message;
        // The payload send, or `nullptr` for a message without payload.
        std::unique_ptr<Communicator::Future> future;
    };

    /// @brief The communication state of a single peer.
    struct PeerState {
        // Received messages waiting for their data receive to be set up, in order.
        std::deque<TagMessage> incoming;
        // Messages whose data is in transit, in order.
        std::deque<TagMessage> in_transit;
        // Retained messages whose send is in flight, in send order.
        std::vector<RetainedMessage> retained_messages;
        // Messages in `incoming` waiting for a receive buffer.
        std::size_t num_awaiting_buffer{0};
        // Application messages sent to the peer.
        std::size_t messages_sent{0};
        // Application messages received from the peer.
        std::size_t messages_received{0};
        // Application messages sent by the peer, known once `terminated` is set.
        std::size_t messages_expected{0};
        // Whether the termination marker of the peer has been received.
        bool terminated{false};
        // DISK payloads handed over by path to the peer.
        std::size_t handoffs_sent{0};
        // DISK payloads handed over by path from the peer.
        std::size_t handoffs_received{0};
        // Whether the peer confirmed that it took over every handed-over file.
        bool handoffs_acked{false};
        // Whether this rank confirmed to the peer that it took over every file.
        bool handoffs_ack_sent{false};

        /**
         * @brief Whether all messages from the peer have been received.
         *
         * @return True if the termination marker and all messages it announced have
         * been received.
         */
        [[nodiscard]] bool received_all() const noexcept {
            return terminated && messages_received >= messages_expected;
        }

        /**
         * @brief Whether the peer still has to confirm it took over the handed-over
         * files.
         *
         * @return True if files were handed over to the peer and it has not confirmed.
         */
        [[nodiscard]] bool awaiting_handoffs_ack() const noexcept {
            return handoffs_sent > 0 && !handoffs_acked;
        }

        /**
         * @brief Whether this rank still has to confirm to the peer that it took over
         * the handed-over files.
         *
         * @return True if the peer handed over files and no confirmation was sent.
         */
        [[nodiscard]] bool owes_handoffs_ack() const noexcept {
            return handoffs_received > 0 && !handoffs_ack_sent;
        }

        /**
         * @brief Whether no message to or from the peer is in progress.
         *
         * @return True if no message is waiting, in transit or retained.
         */
        [[nodiscard]] bool io_idle() const noexcept {
            return incoming.empty() && in_transit.empty() && retained_messages.empty();
        }
    };

    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank const nranks_;
    Rank const rank_;
    Tag const metadata_tag_;
    Tag const gpu_data_tag_;
    AllocateBufferFn allocate_buffer_fn_;
    Duration const allocation_timeout_;
    BufferResource* br_;
    ForcedReceive const forced_receive_;
    // The topology of `br_` at construction, or `nullptr` if none was attached.
    std::shared_ptr<RankTopology const> topology_;

    // The state of each peer.
    std::unordered_map<Rank, PeerState> peers_;

    // Receive buffer allocation state:
    // - `total_awaiting_buffer_`: messages waiting for a buffer, over all peers.
    // - `allocation_stall_start_`: start of the current period without a successful
    //   receive buffer allocation, set while at least one message is waiting.
    // - `next_arrival_`: arrival counter for incoming messages.
    std::size_t total_awaiting_buffer_{0};
    std::optional<Clock::time_point> allocation_stall_start_{};
    std::uint64_t next_arrival_{0};

    // Whether `finish()` has been called. After that, termination markers are
    // exchanged so each peer knows exactly how many application messages to expect,
    // which makes it safe to reuse op_id (see rapidsai/rapidsmpf#927). recv_from is
    // used instead of recv_any to avoid consuming messages from a future collective.
    bool finished_{false};

    /// Sentinel message_id value used to identify protocol-level termination markers.
    static constexpr std::uint64_t termination_sentinel_ =
        std::numeric_limits<std::uint64_t>::max();

    /// Sentinel message_id value of the confirmation that a peer took over every file
    /// handed over to it. Sent once per peer, after all its messages were received.
    static constexpr std::uint64_t handoffs_ack_sentinel_ = termination_sentinel_ - 1;

    // Ongoing "fire-and-forget" operations (non-blocking sends).
    std::vector<std::unique_ptr<Communicator::Future>> fire_and_forget_;
    // Futures corresponding to in-transit messages.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;
    // Completed futures of in-transit messages that wait for an earlier message from
    // the same peer to complete.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        finished_futures_;
    // Messages that have completed and are ready to be retrieved.
    std::vector<std::unique_ptr<Message>> received_messages_;

    // Sent messages ready to be returned by `release_sent()`.
    std::vector<std::unique_ptr<Message>> sent_messages_;

    // Statistics tracking
    std::shared_ptr<Statistics> statistics_;

    // Sequential message ID generator
    std::uint64_t next_message_id_{0};

    /**
     * @brief Send a message whose DISK payload is handed over by file path.
     *
     * Only the metadata message is sent, carrying the path of a file the receiver takes
     * ownership of: the payload's own file, or a clone of it for a retained message.
     *
     * @param message The message, with a non-empty DISK payload.
     * @param message_id The message ID.
     * @param state The state of the destination peer.
     *
     * @throw std::logic_error if the payload has no backing file.
     */
    void send_disk_handoff(
        std::unique_ptr<Message> message, std::uint64_t message_id, PeerState& state
    );

    /**
     * @brief Receive metadata for incoming messages.
     */
    void receive_metadata();

    /**
     * @brief Confirm to each peer that every file it handed over has been taken over.
     *
     * Sent once per peer that handed over files, after all its messages were received
     * and taken out of `incoming`, at which point every handed-over file was adopted.
     */
    void send_handoffs_acks();

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

    /**
     * @brief Move retained messages whose send has completed to `sent_messages_`.
     */
    void complete_retained_messages();
};


}  // namespace rapidsmpf::communicator
