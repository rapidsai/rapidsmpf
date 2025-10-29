/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::communicator {

MetadataPayloadExchange::Message::Message(
    Rank peer_rank, std::vector<std::uint8_t> metadata, std::unique_ptr<Buffer> data
)
    : peer_rank_(peer_rank), metadata_(std::move(metadata)), data_(std::move(data)) {}

Buffer const* MetadataPayloadExchange::Message::data() const {
    return data_.get();
}

std::unique_ptr<Buffer> MetadataPayloadExchange::Message::release_data() {
    return std::move(data_);
}

void MetadataPayloadExchange::Message::set_data(std::unique_ptr<Buffer> buffer) {
    RAPIDSMPF_EXPECTS(data_ == nullptr, "data already set");
    data_ = std::move(buffer);
}


}  // namespace rapidsmpf::communicator
