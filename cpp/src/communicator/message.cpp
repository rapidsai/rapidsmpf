/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/message.hpp>

namespace rapidsmpf::communicator {

Message::Message(
    Rank peer_rank, std::vector<std::uint8_t> metadata, std::unique_ptr<Buffer> data
)
    : peer_rank_(peer_rank), metadata_(std::move(metadata)), data_(std::move(data)) {}

Rank Message::peer_rank() const {
    return peer_rank_;
}

std::vector<std::uint8_t> const& Message::metadata() const {
    return metadata_;
}

Buffer const* Message::data() const {
    return data_.get();
}

std::unique_ptr<Buffer> Message::release_data() {
    return std::move(data_);
}

void Message::set_data(std::unique_ptr<Buffer> buffer) {
    data_ = std::move(buffer);
}

std::uint64_t Message::message_id() const {
    return message_id_;
}

void Message::set_message_id(std::uint64_t id) {
    message_id_ = id;
}

std::size_t Message::expected_payload_size() const {
    return expected_payload_size_;
}

void Message::set_expected_payload_size(std::size_t size) {
    expected_payload_size_ = size;
}

}  // namespace rapidsmpf::communicator
