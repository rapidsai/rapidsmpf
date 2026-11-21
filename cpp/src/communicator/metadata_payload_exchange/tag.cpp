/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

#include <cuda_runtime.h>

#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::communicator {

namespace {

/// Wire `payload_size` of a message whose DISK payload is handed over by file path.
constexpr std::size_t disk_handoff_sentinel = std::numeric_limits<std::size_t>::max();

}  // namespace

void TagMetadataPayloadExchange::TagMessage::adopt_handoff(
    BufferResource& br, Statistics& statistics
) {
    auto buffer = br.adopt_disk_file(handoff_path, br.stream_pool()->get_stream());
    statistics.add_bytes_stat(
        "metadata-payload-exchange-disk-handoff-received", buffer->size
    );
    message->set_data(std::move(buffer));
    handoff_path.clear();
}

/**
 * @brief A message's metadata and protocol trailer.
 *
 * Wire format: [metadata][message_id (8)][payload_size (8)], or for a DISK handoff
 * [metadata][path][path_len (4)][message_id (8)][disk_handoff_sentinel (8)].
 */
struct TagMetadataPayloadExchange::CombinedMetadata {
    std::vector<std::uint8_t> metadata;
    std::uint64_t message_id;
    std::size_t payload_size;  ///< 0 for a handoff.
    std::filesystem::path handoff_path;  ///< Empty unless a handoff.

    CombinedMetadata(
        std::vector<std::uint8_t> metadata,
        std::uint64_t message_id,
        std::size_t payload_size,
        std::filesystem::path handoff_path = {}
    )
        : metadata{std::move(metadata)},
          message_id{message_id},
          payload_size{payload_size},
          handoff_path{std::move(handoff_path)} {}

    [[nodiscard]] bool is_termination_sentinel() const noexcept {
        return message_id == termination_sentinel_;
    }

    /// Serialized size in bytes.
    [[nodiscard]] std::size_t size() const noexcept {
        return metadata.size() + sizeof(message_id) + sizeof(payload_size)
               + (handoff_path.empty()
                      ? 0
                      : handoff_path.native().size() + sizeof(std::uint32_t));
    }

    /// Append the trailer to the metadata.
    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> serialize() && {
        auto const& path = handoff_path.native();
        auto const path_len = safe_cast<std::uint32_t>(path.size());
        std::size_t const wire_payload_size =
            path_len > 0 ? disk_handoff_sentinel : payload_size;

        // Size and offset must be read before `metadata` is moved out.
        std::size_t const total_size = size();
        std::size_t offset = metadata.size();
        auto ret = std::make_unique<std::vector<std::uint8_t>>(std::move(metadata));
        ret->resize(total_size);
        auto write = [&](void const* src, std::size_t size) {
            std::memcpy(ret->data() + offset, src, size);
            offset += size;
        };
        if (path_len > 0) {
            write(path.data(), path.size());
            write(&path_len, sizeof(path_len));
        }
        write(&message_id, sizeof(message_id));
        write(&wire_payload_size, sizeof(wire_payload_size));
        return ret;
    }

    /// Split a received buffer into metadata and trailer, reusing the buffer.
    /// @throw std::logic_error if the buffer is truncated or malformed.
    [[nodiscard]] static CombinedMetadata deserialize(std::vector<std::uint8_t>&& msg) {
        // Read the trailer from the back. Bounds are checked before each `pop_back`.
        std::size_t end = msg.size();
        auto pop_back = [&](auto& value) {
            end -= sizeof(value);
            std::memcpy(&value, msg.data() + end, sizeof(value));
        };

        std::size_t payload_size;
        std::uint64_t message_id;
        RAPIDSMPF_EXPECTS(
            end >= sizeof(payload_size) + sizeof(message_id), "Truncated metadata"
        );
        pop_back(payload_size);
        pop_back(message_id);

        std::filesystem::path handoff_path;
        if (payload_size == disk_handoff_sentinel) {
            std::uint32_t path_len;
            RAPIDSMPF_EXPECTS(end >= sizeof(path_len), "Truncated metadata");
            pop_back(path_len);
            RAPIDSMPF_EXPECTS(
                path_len > 0 && path_len <= end, "Malformed disk handoff metadata"
            );
            end -= path_len;
            handoff_path =
                std::string{reinterpret_cast<char const*>(msg.data() + end), path_len};
            payload_size = 0;  // No data transfer, so no receive buffer.
        }

        msg.resize(end);
        return {std::move(msg), message_id, payload_size, std::move(handoff_path)};
    }
};

// Defined after `CombinedMetadata`, which must be complete here.
TagMetadataPayloadExchange::TagMessage::TagMessage(
    CombinedMetadata&& combined, Rank peer, std::uint64_t arrival_order
)
    : TagMessage{
          std::make_unique<Message>(peer, std::move(combined.metadata), nullptr),
          combined.message_id,
          combined.payload_size,
          arrival_order,
          std::move(combined.handoff_path)
      } {}

TagMetadataPayloadExchange::TagMetadataPayloadExchange(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    std::vector<Rank> peers,
    AllocateBufferFn allocate_buffer_fn,
    Duration allocation_timeout,
    BufferResource* br,
    ForcedReceive forced_receive
)
    : comm_(std::move(comm)),
      nranks_(comm_->nranks()),
      rank_(comm_->rank()),
      metadata_tag_{op_id, 0},
      gpu_data_tag_{op_id, 1},
      allocate_buffer_fn_(std::move(allocate_buffer_fn)),
      allocation_timeout_(allocation_timeout),
      br_(br),
      forced_receive_(forced_receive) {
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be NULL");
    topology_ = br_->rank_topology();
    RAPIDSMPF_EXPECTS(
        topology_ == nullptr
            || (std::cmp_equal(topology_->ranks().size(), nranks_)
                && topology_->local().rank == rank_),
        "the rank topology of the buffer resource does not match the communicator",
        std::invalid_argument
    );
    for (Rank peer : peers) {
        RAPIDSMPF_EXPECTS(
            peer >= 0 && peer < nranks_, "peer is not a valid rank", std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            peer != rank_, "peers cannot include the calling rank", std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            peers_.try_emplace(peer).second,
            "peers cannot contain duplicates",
            std::invalid_argument
        );
    }
    statistics_ = br_->statistics();
}

TagMetadataPayloadExchange::~TagMetadataPayloadExchange() noexcept {
    // Files handed over to this rank but never adopted are owned by nobody else.
    for (auto& [_, state] : peers_) {
        for (auto const& tag_msg : state.incoming) {
            if (!tag_msg.handoff_path.empty()) {
                std::error_code ignored;
                std::filesystem::remove(tag_msg.handoff_path, ignored);
            }
        }
    }
    if (finished_ && TagMetadataPayloadExchange::is_idle()) {
        return;
    }
    try {
        comm_->logger()->warn(
            "destroying a TagMetadataPayloadExchange that is not finished and idle (",
            finished_ ? "finished" : "not finished",
            ", sends in flight: ",
            fire_and_forget_.size(),
            ", receives in flight: ",
            in_transit_futures_.size() + finished_futures_.size(),
            ", uncollected received messages: ",
            received_messages_.size(),
            ", unreleased sent messages: ",
            sent_messages_.size(),
            ")"
        );
    } catch (...) {
        // Ignore logging failures.
    }
}

std::span<MemoryType const> TagMetadataPayloadExchange::sendable_memory_types(
    Rank dst
) const {
    return topology_ != nullptr && topology_->can_read_spill_files(dst)
               ? std::span<MemoryType const>{MEMORY_TYPES}
               : ADDRESSABLE_MEMORY_TYPES;
}

void TagMetadataPayloadExchange::send(
    std::unique_ptr<MetadataPayloadExchange::Message> message
) {
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    messages.push_back(std::move(message));
    send(std::move(messages));
}

void TagMetadataPayloadExchange::send(
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>>&& messages
) {
    auto& log = *comm_->logger();
    auto const t0 = Clock::now();

    RAPIDSMPF_EXPECTS(!finished_, "send() called after finish()");

    for (auto const& message : messages) {
        auto const dst = message->peer_rank();
        RAPIDSMPF_EXPECTS(
            peers_.contains(dst), "the destination is not a peer", std::invalid_argument
        );
        auto msg_data = message->data();
        if (msg_data != nullptr) {
            RAPIDSMPF_EXPECTS(
                contains(sendable_memory_types(dst), msg_data->mem_type()),
                "the payload's memory type is not sendable to its destination",
                std::invalid_argument
            );
            // The topology describes the files of `br_`'s disk resource only.
            RAPIDSMPF_EXPECTS(
                msg_data->mem_type() != MemoryType::DISK
                    || msg_data->get_storage<Buffer::DiskBufferT>()->disk_resource()
                           == br_->disk_resource(),
                "a DISK payload must belong to the disk resource of the exchange",
                std::invalid_argument
            );
        }
    }

    // Send metadata followed immediately by data for each message
    for (auto&& message : messages) {
        auto dst = message->peer_rank();
        auto msg_data = message->data();
        auto& state = peers_.at(dst);
        state.messages_sent++;

        // Assign sequential message ID
        // Format: [rank (32 bits)][sequence (32 bits)]
        std::uint64_t message_id =
            (safe_cast<std::uint64_t>(comm_->rank()) << 32) | next_message_id_++;

        log.trace("send metadata to ", dst, " (message_id=", message_id, ")");

        std::size_t payload_size = (msg_data != nullptr) ? msg_data->size : 0;

        // A DISK payload is handed over by path, without sending its bytes.
        if (payload_size > 0 && msg_data->mem_type() == MemoryType::DISK) {
            send_disk_handoff(std::move(message), message_id, state);
            continue;
        }

        // A retained message keeps its metadata, so it is copied. Otherwise, it is
        // released to avoid the copy.
        fire_and_forget_.push_back(comm_->send(
            CombinedMetadata{
                message->after_send() == AfterSend::RETAIN ? message->metadata()
                                                           : message->release_metadata(),
                message_id,
                payload_size
            }
                .serialize(),
            dst,
            metadata_tag_
        ));

        // Send data immediately after metadata (if any)
        std::unique_ptr<Communicator::Future> payload_future;
        if (payload_size > 0) {
            payload_future = comm_->send(message->release_data(), dst, gpu_data_tag_);
        }
        if (message->after_send() == AfterSend::RETAIN) {
            state.retained_messages.push_back(
                {std::move(message), std::move(payload_future)}
            );
        } else if (payload_future) {
            fire_and_forget_.push_back(std::move(payload_future));
        }
    }

    statistics_->add_duration_stat(
        "metadata-payload-exchange-send-messages", Clock::now() - t0
    );
}

void TagMetadataPayloadExchange::send_disk_handoff(
    std::unique_ptr<MetadataPayloadExchange::Message> message,
    std::uint64_t message_id,
    PeerState& state
) {
    auto const dst = message->peer_rank();
    auto const* data = message->data();
    std::size_t const payload_size = data->size;
    auto const& disk = *data->get_storage<Buffer::DiskBufferT>();

    auto do_fire_and_forget = [&](std::filesystem::path path,
                                  std::vector<std::uint8_t> metadata) {
        RAPIDSMPF_EXPECTS(
            !path.empty(), "the DISK payload has no backing file", std::logic_error
        );
        comm_->logger()->trace(
            "hand over ", path, " to ", dst, " (message_id=", message_id, ")"
        );
        fire_and_forget_.push_back(comm_->send(
            CombinedMetadata{std::move(metadata), message_id, 0, std::move(path)}
                .serialize(),
            dst,
            metadata_tag_
        ));
        statistics_->add_bytes_stat(
            "metadata-payload-exchange-disk-handoff-sent", payload_size
        );
    };


    // The receiver takes ownership of the handed-over file: a retained message keeps its
    // own file and hands over a clone, otherwise the file itself is handed over.
    switch (message->after_send()) {
    case AfterSend::RETAIN:
        {
            do_fire_and_forget(
                disk.disk_resource()->clone(disk.path(), data->stream()),
                message->metadata()
            );
            state.retained_messages.push_back({std::move(message), nullptr});
            break;
        }
    case AfterSend::DISCARD:
        do_fire_and_forget(
            BufferResource::release_disk_file(message->release_data()),
            message->release_metadata()
        );
        break;
    }
}

void TagMetadataPayloadExchange::progress() {
    auto const t0 = Clock::now();

    // Process all phases of the communication protocol
    receive_metadata();
    auto completed_metadata_only = setup_data_receives();
    auto completed_data = complete_data_transfers();

    // Store all completed messages
    std::ranges::move(completed_metadata_only, std::back_inserter(received_messages_));
    std::ranges::move(completed_data, std::back_inserter(received_messages_));

    cleanup_completed_operations();
    complete_retained_messages();

    statistics_->add_duration_stat(
        "metadata-payload-exchange-progress", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MetadataPayloadExchange::Message>>
TagMetadataPayloadExchange::recv() {
    // Move all completed messages to the caller
    return std::move(received_messages_);
}

std::vector<std::unique_ptr<MetadataPayloadExchange::Message>>
TagMetadataPayloadExchange::release_sent() {
    return std::move(sent_messages_);
}

void TagMetadataPayloadExchange::finish() {
    if (finished_) {
        return;
    }
    finished_ = true;

    // Send a termination marker to every peer. The marker tells each peer
    // exactly how many application messages we sent to it, so the peer can
    // stop receiving once it has them all.
    // The marker has no metadata, the sentinel as message ID and the message count as
    // payload size.
    for (auto const& [peer, state] : peers_) {
        fire_and_forget_.push_back(comm_->send(
            CombinedMetadata{{}, termination_sentinel_, state.messages_sent}.serialize(),
            peer,
            metadata_tag_
        ));
    }
}

bool TagMetadataPayloadExchange::is_finished() const {
    return finished_;
}

bool TagMetadataPayloadExchange::is_idle() const {
    if (!fire_and_forget_.empty() || !in_transit_futures_.empty()
        || !finished_futures_.empty() || !received_messages_.empty()
        || !sent_messages_.empty())
    {
        return false;
    }
    // After finish(), also require all peers to have sent their termination
    // markers and all expected application messages to have been received.
    return std::ranges::all_of(peers_, [&](auto const& peer) {
        return peer.second.io_idle() && (!finished_ || peer.second.received_all());
    });
}

void TagMetadataPayloadExchange::receive_metadata() {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    // Use per-peer recv_from to avoid consuming messages belonging to a future
    // collective on the same tag (see rapidsai/rapidsmpf#927).
    for (auto& [peer, state] : peers_) {
        // Skip peers that are fully done.
        if (state.received_all()) {
            continue;
        }

        while (true) {
            auto msg = comm_->recv_from(peer, metadata_tag_);
            if (!msg) {
                break;
            }

            auto combined = CombinedMetadata::deserialize(std::move(*msg));
            auto const& payload_size = combined.payload_size;

            // Check for termination marker.
            if (combined.is_termination_sentinel()) {
                state.messages_expected = payload_size;  // payload_size carries the count
                state.terminated = true;
                log->trace(
                    "recv termination from ",
                    peer,
                    " (expected_messages=",
                    payload_size,
                    ")"
                );
                break;
            }

            // Normal application message.
            state.messages_received++;

            log->trace("recv_from ", peer, " (message_id=", combined.message_id, ")");
            auto& tag_msg =
                state.incoming.emplace_back(std::move(combined), peer, next_arrival_++);
            if (payload_size > 0) {
                // Buffers are allocated in message order per peer, so only try to
                // allocate when no earlier message from this peer is still waiting.
                state.num_awaiting_buffer++;
                total_awaiting_buffer_++;
                if (state.num_awaiting_buffer == 1) {
                    try_attach_receive_buffer(tag_msg, AllocationMode::MAY_FAIL);
                }
                if (tag_msg.is_awaiting_buffer() && !allocation_stall_start_.has_value())
                {
                    allocation_stall_start_ = Clock::now();
                }
            }
        }
    }

    statistics_->add_duration_stat(
        "metadata-payload-exchange-receive-metadata", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MetadataPayloadExchange::Message>>
TagMetadataPayloadExchange::setup_data_receives() {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> completed_messages;

    force_allocation_progress_if_stalled();

    // Process messages per rank, breaking only when a rank's buffer isn't ready
    for (auto& [src, state] : peers_) {
        auto& messages = state.incoming;

        // Process messages for this rank in order
        auto msg_it = messages.begin();
        while (msg_it != messages.end()) {
            auto& tag_msg = *msg_it;
            log->trace(
                "checking incoming message data from ",
                src,
                " (message_id=",
                tag_msg.message_id,
                ")"
            );

            std::size_t payload_size = tag_msg.expected_payload_size;

            if (payload_size > 0) {
                // Allocate the buffer if this message is still waiting for one. If that
                // fails, break for this rank and retry on a later call.
                if (tag_msg.is_awaiting_buffer()
                    && !try_attach_receive_buffer(tag_msg, AllocationMode::MAY_FAIL))
                {
                    break;
                }

                // Check if the buffer is ready for use, if not, break for this rank
                // and wait for the buffer to be ready. This is necessary to ensure
                // messages are received in the order they are sent from this rank.
                if (!tag_msg.message->data()->is_latest_write_done()) {
                    break;
                }

                // Extract the message and set up for data transfer
                auto tag_message = std::move(tag_msg);
                msg_it = messages.erase(msg_it);

                auto data_buffer = tag_message.message->release_data();
                auto future = comm_->recv(src, gpu_data_tag_, std::move(data_buffer));

                auto message_id = tag_message.message_id;
                RAPIDSMPF_EXPECTS(
                    in_transit_futures_.emplace(message_id, std::move(future)).second,
                    "in transit future already exists"
                );
                // Store in per-rank vector to maintain order
                state.in_transit.push_back(std::move(tag_message));
                // Break to ensure we don't return later messages before this one
                // completes
                break;
            } else {
                // Control/metadata-only message
                // Only return if there are no earlier in-transit messages from this rank
                if (state.in_transit.empty()) {
                    if (!tag_msg.handoff_path.empty()) {
                        tag_msg.adopt_handoff(*br_, *statistics_);
                    }
                    completed_messages.push_back(std::move(tag_msg.message));
                    msg_it = messages.erase(msg_it);
                } else {
                    // There are earlier messages still in transit, stop processing this
                    // rank
                    break;
                }
            }
        }
    }

    statistics_->add_duration_stat(
        "metadata-payload-exchange-setup-data-receives", Clock::now() - t0
    );

    return completed_messages;
}

std::vector<std::unique_ptr<MetadataPayloadExchange::Message>>
TagMetadataPayloadExchange::complete_data_transfers() {
    auto const t0 = Clock::now();

    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> completed_messages;

    // Move completed futures out of `in_transit_futures_`.
    if (!in_transit_futures_.empty()) {
        finished_futures_.merge(comm_->test_some(in_transit_futures_));
    }

    // Return messages in order per peer, stopping at the first incomplete one.
    if (!finished_futures_.empty()) {
        for (auto& [src, state] : peers_) {
            auto& messages = state.in_transit;
            while (!messages.empty()) {
                auto& tag_msg = messages.front();
                auto node = finished_futures_.extract(tag_msg.message_id);
                if (node.empty()) {
                    break;
                }
                auto received_buffer = comm_->release_data(std::move(node.mapped()));
                if (tag_msg.forced_allocation
                    && forced_receive_ == ForcedReceive::MOVE_TO_DISK)
                {
                    try_move_to_disk(received_buffer);
                }
                tag_msg.message->set_data(std::move(received_buffer));
                completed_messages.push_back(std::move(tag_msg.message));
                messages.pop_front();
            }
        }
    }

    statistics_->add_duration_stat(
        "metadata-payload-exchange-complete-data-transfers", Clock::now() - t0
    );

    return completed_messages;
}

bool TagMetadataPayloadExchange::try_attach_receive_buffer(
    TagMessage& tag_msg, AllocationMode mode
) {
    RAPIDSMPF_EXPECTS(
        tag_msg.is_awaiting_buffer(), "message is not waiting for a buffer"
    );
    auto buffer = allocate_buffer_fn_(br_, tag_msg.expected_payload_size, mode);
    if (buffer == nullptr) {
        RAPIDSMPF_EXPECTS(
            mode == AllocationMode::MAY_FAIL,
            "receive buffer allocator returned nullptr although it must succeed",
            std::runtime_error
        );
        return false;
    }
    tag_msg.message->set_data(std::move(buffer));

    peers_.at(tag_msg.message->peer_rank()).num_awaiting_buffer--;
    total_awaiting_buffer_--;

    // An allocation succeeded, so restart the stall period if messages still wait.
    if (total_awaiting_buffer_ > 0) {
        allocation_stall_start_ = Clock::now();
    } else {
        allocation_stall_start_.reset();
    }
    return true;
}

void TagMetadataPayloadExchange::force_allocation_progress_if_stalled() {
    if (!allocation_stall_start_.has_value()
        || Clock::now() - *allocation_stall_start_ < allocation_timeout_)
    {
        return;
    }
    // Find the oldest message waiting for a buffer. Per peer, that is the first
    // waiting message, since buffers are allocated in message order.
    TagMessage* oldest = nullptr;
    for (auto& [_, state] : peers_) {
        auto it = std::ranges::find_if(state.incoming, &TagMessage::is_awaiting_buffer);
        if (it != state.incoming.end()
            && (oldest == nullptr || it->arrival < oldest->arrival))
        {
            oldest = &*it;
        }
    }
    RAPIDSMPF_EXPECTS(oldest != nullptr, "no message is waiting for a buffer");
    // The oldest message may not have been tried since the last allocation succeeded,
    // for example when it waited behind an earlier message from the same peer. Only
    // force it if a normal allocation fails.
    if (try_attach_receive_buffer(*oldest, AllocationMode::MAY_FAIL)) {
        return;
    }
    try_attach_receive_buffer(*oldest, AllocationMode::MUST_SUCCEED);
    statistics_->add_stat("metadata-payload-exchange-forced-allocations", 1);
    oldest->forced_allocation = true;
}

void TagMetadataPayloadExchange::try_move_to_disk(std::unique_ptr<Buffer>& buffer) {
    if (br_->disk_resource() == nullptr) {
        return;
    }
    auto reservation = br_->try_reserve(buffer->size, MemoryType::DISK);
    if (!reservation.has_value()) {
        return;
    }
    statistics_->add_bytes_stat("metadata-payload-exchange-moved-to-disk", buffer->size);
    buffer = br_->move(std::move(buffer), *reservation);
}

void TagMetadataPayloadExchange::cleanup_completed_operations() {
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }
}

void TagMetadataPayloadExchange::complete_retained_messages() {
    for (auto& [_, state] : peers_) {
        auto& retained = state.retained_messages;
        // Release messages in send order, stopping at the first incomplete send.
        auto it = retained.begin();
        for (; it != retained.end(); ++it) {
            if (it->future != nullptr) {
                if (!comm_->test(it->future)) {
                    break;
                }
                it->message->set_data(comm_->release_data(std::move(it->future)));
            }
            sent_messages_.push_back(std::move(it->message));
        }
        retained.erase(retained.begin(), it);
    }
}

}  // namespace rapidsmpf::communicator
