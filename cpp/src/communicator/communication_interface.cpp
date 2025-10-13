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
      ready_for_data_tag_{op_id, 1},
      metadata_tag_{op_id, 2},
      gpu_data_tag_{op_id, 3},
      statistics_{std::move(statistics)} {}

void TagCommunicationInterface::submit_outgoing_messages(
    std::vector<std::unique_ptr<MessageInterface>>&& messages, BufferResource* br
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    // Store messages for sending and initiate metadata transmission
    for (auto&& message : messages) {
        auto dst = message->peer_rank();
        log.trace("send metadata to ", dst, ": ", message->to_string());
        RAPIDSMPF_EXPECTS(dst != rank_, "sending message to ourselves");

        auto metadata = message->serialize_metadata();
        fire_and_forget_.push_back(comm_->send(
            std::make_unique<std::vector<std::uint8_t>>(std::move(metadata)),
            dst,
            metadata_tag_
        ));

        if (message->total_data_size() > 0) {
            auto message_id = message->message_id();
            RAPIDSMPF_EXPECTS(
                outgoing_messages_.emplace(message_id, std::move(message)).second,
                "outgoing message already exists"
            );
            ready_ack_receives_[dst].push_back(comm_->recv_sync_host_data(
                dst,
                ready_for_data_tag_,
                std::make_unique<std::vector<uint8_t>>(ReadyForDataMessage::byte_size)
            ));
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-submit-outgoing-messages", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MessageInterface>>
TagCommunicationInterface::process_communication(MessageFactory const& message_factory) {
    auto const t0 = Clock::now();

    // Process all phases of the communication protocol
    receive_metadata(message_factory);
    setup_data_receives(message_factory);
    process_ready_acks();
    auto completed_messages = complete_data_transfers();
    cleanup_completed_operations();

    statistics_->add_duration_stat(
        "comms-interface-process-communication-total", Clock::now() - t0
    );

    return completed_messages;
}

bool TagCommunicationInterface::is_idle() const {
    return fire_and_forget_.empty() && incoming_messages_.empty()
           && outgoing_messages_.empty() && in_transit_messages_.empty()
           && in_transit_futures_.empty()
           && std::ranges::all_of(ready_ack_receives_, [](auto const& kv) {
                  return kv.second.empty();
              });
}

void TagCommunicationInterface::receive_metadata(MessageFactory const& message_factory) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag_);
        if (!msg)
            break;

        // The msg is already a vector<uint8_t>, so we can use it directly
        auto message = message_factory.create_from_metadata(*msg, src);
        log.trace("recv_any from ", src, ": ", message->to_string());
        incoming_messages_.emplace(src, std::move(message));
    }

    statistics_->add_duration_stat("comms-interface-receive-metadata", Clock::now() - t0);
}

void TagCommunicationInterface::setup_data_receives(
    MessageFactory const& message_factory
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        log.trace(
            "checking incoming message data from ", src, ": ", message->to_string()
        );

        if (message->total_data_size() > 0) {
            if (!message->is_data_ready()) {
                auto buffer = message_factory.allocate_receive_buffer(
                    message->total_data_size(), *message
                );
                message->set_data_buffer(std::move(buffer));
            }

            if (!message->is_ready()) {
                ++it;
                continue;
            }

            // Extract the message and set up for data transfer
            auto message_ptr = std::move(it->second);
            auto src = it->first;
            it = incoming_messages_.erase(it);

            auto data_buffer = message_ptr->release_data_buffer();
            RAPIDSMPF_EXPECTS(data_buffer, "No data buffer available");
            auto future = comm_->recv(src, gpu_data_tag_, std::move(data_buffer));

            auto message_id = message_ptr->message_id();
            RAPIDSMPF_EXPECTS(
                in_transit_futures_.emplace(message_id, std::move(future)).second,
                "in transit future already exists"
            );
            RAPIDSMPF_EXPECTS(
                in_transit_messages_.emplace(message_id, std::move(message_ptr)).second,
                "in transit message already exists"
            );

            auto ready_msg = ReadyForDataMessage{message_id};
            fire_and_forget_.push_back(comm_->send(
                std::make_unique<std::vector<std::uint8_t>>(ready_msg.pack()),
                src,
                ready_for_data_tag_
            ));
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

void TagCommunicationInterface::process_ready_acks() {
    auto const t0 = Clock::now();

    for (auto& [dst, futures] : ready_ack_receives_) {
        auto [finished, _] = comm_->test_some(futures);
        for (auto&& future : finished) {
            auto const msg_data = comm_->release_sync_host_data(std::move(future));
            auto msg = ReadyForDataMessage::unpack(msg_data);

            auto message_it = outgoing_messages_.find(msg.message_id);
            RAPIDSMPF_EXPECTS(
                message_it != outgoing_messages_.end(), "outgoing message not found"
            );

            auto data_buffer = message_it->second->release_data_buffer();
            RAPIDSMPF_EXPECTS(data_buffer, "No data buffer available");

            fire_and_forget_.push_back(
                comm_->send(std::move(data_buffer), dst, gpu_data_tag_)
            );

            outgoing_messages_.erase(message_it);
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-process-ready-acks", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MessageInterface>>
TagCommunicationInterface::complete_data_transfers() {
    auto const t0 = Clock::now();

    std::vector<std::unique_ptr<MessageInterface>> completed_messages;

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

            message->set_data_buffer(std::move(received_buffer));

            completed_messages.push_back(std::move(message));

            in_transit_messages_.erase(message_it);
            in_transit_futures_.erase(future_it);
        }
    }

    // Handle control/metadata-only messages from incoming_messages_
    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        if (message->total_data_size() == 0) {
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

std::vector<std::uint8_t> ReadyForDataMessage::pack() const {
    std::vector<std::uint8_t> buffer(byte_size);
    std::memcpy(buffer.data(), &message_id, sizeof(message_id));
    return buffer;
}

ReadyForDataMessage ReadyForDataMessage::unpack(std::vector<std::uint8_t> const& data) {
    RAPIDSMPF_EXPECTS(data.size() == byte_size, "Invalid message size");
    ReadyForDataMessage msg;
    std::memcpy(&msg.message_id, data.data(), sizeof(msg.message_id));
    return msg;
}

std::string ReadyForDataMessage::to_string() const {
    return "ReadyForDataMessage{message_id=" + std::to_string(message_id) + "}";
}

}  // namespace rapidsmpf::communicator
