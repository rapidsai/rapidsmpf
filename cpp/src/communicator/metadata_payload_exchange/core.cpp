/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::communicator {

MetadataPayloadExchange::Message::Message(
    Rank peer_rank, std::vector<std::uint8_t>&& metadata, std::unique_ptr<Buffer> data
)
    : peer_rank_(peer_rank), metadata_(std::move(metadata)), data_(std::move(data)) {}

Buffer const* MetadataPayloadExchange::Message::data() const {
    return data_.get();
}

std::vector<std::uint8_t> MetadataPayloadExchange::Message::release_metadata() noexcept {
    return std::move(metadata_);
}

std::unique_ptr<Buffer> MetadataPayloadExchange::Message::release_data() noexcept {
    return std::move(data_);
}

void MetadataPayloadExchange::Message::set_data(std::unique_ptr<Buffer> buffer) {
    RAPIDSMPF_EXPECTS(data_ == nullptr, "data already set");
    data_ = std::move(buffer);
}

std::vector<Rank> all_remote_ranks(Communicator const& comm) {
    std::vector<Rank> ret;
    ret.reserve(safe_cast<std::size_t>(comm.nranks()));
    for (Rank rank = 0; rank < comm.nranks(); ++rank) {
        if (rank != comm.rank()) {
            ret.push_back(rank);
        }
    }
    return ret;
}


}  // namespace rapidsmpf::communicator
