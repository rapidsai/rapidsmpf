/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::communicator {

MetadataPayloadExchange::Message::Message(
    Rank peer_rank,
    std::vector<std::uint8_t>&& metadata,
    std::unique_ptr<Buffer> data,
    ReturnPayload return_payload
)
    : peer_rank_(peer_rank),
      metadata_(std::move(metadata)),
      data_(std::move(data)),
      return_payload_(return_payload) {}

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

std::unique_ptr<Buffer> default_allocate_buffer(
    BufferResource* br, std::size_t size, MetadataPayloadExchange::AllocationMode mode
) {
    auto res = br->try_reserve(size, ADDRESSABLE_MEMORY_TYPES);
    if (!res.has_value()) {
        if (mode == MetadataPayloadExchange::AllocationMode::MAY_FAIL) {
            return nullptr;
        }
        RAPIDSMPF_EXPECTS(
            br->allow_overbooking_by_default() == AllowOverbooking::YES,
            "receive buffer allocation made no progress within the allocation timeout "
            "and overbooking is not allowed",
            std::runtime_error
        );
        // Fall back to overbooking host memory as a last resort.
        res = br->reserve(MemoryType::HOST, size, AllowOverbooking::YES).first;
    }
    return br->make_buffer(br->stream_pool()->get_stream(), std::move(*res));
}

}  // namespace rapidsmpf::communicator
