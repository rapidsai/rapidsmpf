/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

#include <cuda_runtime.h>

#include <rapidsmpf/communicator/communication_interface.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::communicator {


TagCommunicationInterface::TagCommunicationInterface(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    Rank rank,
    std::shared_ptr<Statistics> statistics
)
    : comm_(std::move(comm)),
      rank_(rank),
      metadata_tag_{op_id, 1},
      gpu_data_tag_{op_id, 2},
      statistics_{std::move(statistics)} {}

void TagCommunicationInterface::submit_outgoing_messages(
    std::vector<std::unique_ptr<Message>>&& messages
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    // Send metadata followed immediately by data for each message
    for (auto&& message : messages) {
        auto dst = message->peer_rank();

        // Assign sequential message ID (unique per rank)
        // Format: [rank (32 bits)][sequence (32 bits)]
        std::uint64_t message_id =
            (static_cast<std::uint64_t>(rank_) << 32) | next_message_id_++;
        message->set_message_id(message_id);

        log.trace("send metadata to ", dst, " (message_id=", message_id, ")");
        RAPIDSMPF_EXPECTS(dst != rank_, "sending message to ourselves");

        auto const& original_metadata = message->metadata();
        std::size_t payload_size =
            (message->data() != nullptr) ? message->data()->size : 0;

        // Pack metadata: [message_id][payload_size][original_metadata]
        auto combined_metadata = std::make_unique<std::vector<std::uint8_t>>(
            sizeof(std::uint64_t) + sizeof(std::size_t) + original_metadata.size()
        );

        std::size_t offset = 0;

        std::memcpy(
            combined_metadata->data() + offset, &message_id, sizeof(std::uint64_t)
        );
        offset += sizeof(std::uint64_t);

        std::memcpy(
            combined_metadata->data() + offset, &payload_size, sizeof(std::size_t)
        );
        offset += sizeof(std::size_t);

        std::memcpy(
            combined_metadata->data() + offset,
            original_metadata.data(),
            original_metadata.size()
        );

        fire_and_forget_.push_back(
            comm_->send(std::move(combined_metadata), dst, metadata_tag_)
        );

        // Send data immediately after metadata (if any)
        if (message->data() != nullptr && message->data()->size > 0) {
            auto data_buffer = message->release_data();
            RAPIDSMPF_EXPECTS(data_buffer, "No data buffer available");

            fire_and_forget_.push_back(
                comm_->send(std::move(data_buffer), dst, gpu_data_tag_)
            );
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-submit-outgoing-messages", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<Message>> TagCommunicationInterface::process_communication(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
) {
    auto const t0 = Clock::now();

    // Process all phases of the communication protocol
    receive_metadata();
    setup_data_receives(allocate_buffer_fn);
    auto completed_messages = complete_data_transfers();
    cleanup_completed_operations();

    statistics_->add_duration_stat(
        "comms-interface-process-communication-total", Clock::now() - t0
    );

    return completed_messages;
}

bool TagCommunicationInterface::is_idle() const {
    return fire_and_forget_.empty() && incoming_messages_.empty()
           && in_transit_messages_.empty() && in_transit_futures_.empty();
}

void TagCommunicationInterface::receive_metadata() {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag_);
        if (!msg)
            break;

        // Unpack metadata: [message_id][payload_size][original_metadata]
        if (msg->size() < sizeof(std::uint64_t) + sizeof(std::size_t)) {
            log.warn("Received metadata too small, skipping");
            continue;
        }

        std::size_t offset = 0;

        // Extract message ID
        std::uint64_t message_id;
        std::memcpy(&message_id, msg->data() + offset, sizeof(std::uint64_t));
        offset += sizeof(std::uint64_t);

        // Extract payload size
        std::size_t payload_size;
        std::memcpy(&payload_size, msg->data() + offset, sizeof(std::size_t));
        offset += sizeof(std::size_t);

        // Extract original metadata (everything after message ID and payload size)
        std::vector<std::uint8_t> original_metadata(
            msg->begin() + static_cast<std::ptrdiff_t>(offset), msg->end()
        );

        auto message =
            std::make_unique<Message>(src, std::move(original_metadata), nullptr);

        // Set the message ID and payload size
        message->set_message_id(message_id);
        message->set_expected_payload_size(payload_size);

        log.trace("recv_any from ", src, " (message_id=", message_id, ")");
        incoming_messages_.emplace(src, std::move(message));
    }

    statistics_->add_duration_stat("comms-interface-receive-metadata", Clock::now() - t0);
}

void TagCommunicationInterface::setup_data_receives(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        log.trace(
            "checking incoming message data from ",
            src,
            " (message_id=",
            message->message_id(),
            ")"
        );

        // Get payload size that was extracted during metadata receive
        std::size_t payload_size = message->expected_payload_size();

        if (payload_size > 0) {
            if (message->data() == nullptr) {
                auto buffer = allocate_buffer_fn(payload_size);
                message->set_data(std::move(buffer));
            }

            // Check if the buffer is ready for use
            if (message->data() && !message->data()->is_latest_write_done()) {
                ++it;
                break;
            }

            // Extract the message and set up for data transfer
            auto message_ptr = std::move(it->second);
            auto src_rank = it->first;
            it = incoming_messages_.erase(it);

            auto data_buffer = message_ptr->release_data();
            RAPIDSMPF_EXPECTS(data_buffer, "No data buffer available");
            auto future = comm_->recv(src_rank, gpu_data_tag_, std::move(data_buffer));

            auto message_id = message_ptr->message_id();
            RAPIDSMPF_EXPECTS(
                in_transit_futures_.emplace(message_id, std::move(future)).second,
                "in transit future already exists"
            );
            RAPIDSMPF_EXPECTS(
                in_transit_messages_.emplace(message_id, std::move(message_ptr)).second,
                "in transit message already exists"
            );
        } else {
            // Control/metadata-only message - will be handled in
            // complete_data_transfers()
            ++it;
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-setup-data-receives", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<Message>>
TagCommunicationInterface::complete_data_transfers() {
    auto const t0 = Clock::now();

    std::vector<std::unique_ptr<Message>> completed_messages;

    // Handle completed data transfers
    if (!in_transit_futures_.empty()) {
        std::vector<std::uint64_t> finished = comm_->test_some(in_transit_futures_);
        for (auto message_id : finished) {
            auto message_it = in_transit_messages_.find(message_id);
            auto future_it = in_transit_futures_.find(message_id);

            RAPIDSMPF_EXPECTS(
                message_it != in_transit_messages_.end(), "in transit message not found"
            );
            RAPIDSMPF_EXPECTS(
                future_it != in_transit_futures_.end(), "in transit future not found"
            );

            auto message = std::move(message_it->second);
            auto future = std::move(future_it->second);
            auto received_buffer = comm_->release_data(std::move(future));

            message->set_data(std::move(received_buffer));

            completed_messages.push_back(std::move(message));

            in_transit_messages_.erase(message_it);
            in_transit_futures_.erase(future_it);
        }
    }

    // Handle control/metadata-only messages from incoming_messages_
    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        std::size_t payload_size = message->expected_payload_size();
        if (payload_size == 0) {
            completed_messages.push_back(std::move(it->second));
            it = incoming_messages_.erase(it);
        } else {
            ++it;
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-complete-data-transfers", Clock::now() - t0
    );

    return completed_messages;
}

void TagCommunicationInterface::cleanup_completed_operations() {
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }
}


}  // namespace rapidsmpf::communicator
