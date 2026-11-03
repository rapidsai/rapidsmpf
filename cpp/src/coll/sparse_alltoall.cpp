/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::coll {

SparseAlltoall::SparseAlltoall(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    BufferResource* br,
    std::vector<Rank> srcs,
    std::vector<Rank> dsts,
    std::function<void()>&& finished_callback
)
    : comm_{std::move(comm)},
      br_{br},
      srcs_{std::move(srcs)},
      dsts_{std::move(dsts)},
      finished_callback_{std::move(finished_callback)} {
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be null");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be null");
    auto const size = comm_->nranks();
    auto const self = comm_->rank();
    received_.reserve(srcs_.size());
    for (auto src : srcs_) {
        RAPIDSMPF_EXPECTS(
            src >= 0 && src < size && src != self,
            "SparseAlltoall invalid source rank.",
            std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            received_.try_emplace(src).second,
            "SparseAlltoall source rank list must be unique",
            std::invalid_argument
        );
    }
    next_ordinal_per_dst_.reserve(dsts_.size());
    for (auto dst : dsts_) {
        RAPIDSMPF_EXPECTS(
            dst >= 0 && dst < size && dst != self,
            "SparseAlltoall invalid destination rank.",
            std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            next_ordinal_per_dst_.emplace(dst, 0).second,
            "SparseAlltoall destination rank list must be unique",
            std::invalid_argument
        );
    }
    // SparseAlltoall isn't movable, so the spill function can capture `this`.
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );
    // The peers are the union of the sources and the destinations.
    std::set<Rank> peers{srcs_.begin(), srcs_.end()};
    peers.insert(dsts_.begin(), dsts_.end());
    exchange_ = std::make_unique<communicator::TagMetadataPayloadExchange>(
        comm_,
        op_id,
        std::vector<Rank>(peers.begin(), peers.end()),
        communicator::default_allocate_buffer,
        br_->memory_reserve_timeout(),
        br_
    );
    function_id_ =
        comm_->progress_thread()->add_function([this]() { return event_loop(); });
}

SparseAlltoall::~SparseAlltoall() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        locally_finished_.load(std::memory_order_acquire),
        "Destroying SparseAlltoall without `insert_finished()`"
    );
    br_->spill_manager().remove_spill_function(spill_function_id_);
    comm_->progress_thread()->remove_function(function_id_);
}

std::shared_ptr<Communicator> const& SparseAlltoall::comm() const noexcept {
    return comm_;
}

void SparseAlltoall::insert(Rank dst, PackedData&& packed_data) {
    RAPIDSMPF_EXPECTS(
        !locally_finished_.load(std::memory_order_acquire),
        "SparseAlltoall::insert cannot be called after insert_finished()"
    );
    auto ordinal_it = next_ordinal_per_dst_.find(dst);
    RAPIDSMPF_EXPECTS(
        ordinal_it != next_ordinal_per_dst_.end(),
        "SparseAlltoall::insert destination is not a valid destination"
    );
    packed_data.data = br_->ensure_memory_type(
        std::move(packed_data.data), exchange_->sendable_memory_types(dst)
    );
    outgoing_.insert(
        detail::Chunk::from_packed_data(
            ordinal_it->second.fetch_add(1, std::memory_order_acq_rel),
            comm_->rank(),
            dst,
            std::move(packed_data)
        )
    );
}

void SparseAlltoall::insert_finished() {
    RAPIDSMPF_EXPECTS(
        !locally_finished_.exchange(true, std::memory_order_acq_rel),
        "SparseAlltoall::insert_finished can only be called once"
    );
}

void SparseAlltoall::wait(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (timeout < std::chrono::milliseconds{0}) {
        cv_.wait(lock, [&]() { return can_extract_; });
    } else {
        RAPIDSMPF_EXPECTS(
            cv_.wait_for(lock, timeout, [&]() { return can_extract_; }),
            "SparseAlltoall::wait timeout reached",
            std::runtime_error
        );
    }
}

std::vector<PackedData> SparseAlltoall::extract(Rank src) {
    auto it = received_.find(src);
    RAPIDSMPF_EXPECTS(
        it != received_.end(), "SparseAlltoall::extract provided source is not valid"
    );
    {
        std::lock_guard lock{mutex_};
        RAPIDSMPF_EXPECTS(can_extract_, "Extracting before all chunks are ready");
    }
    auto chunks = it->second.extract();
    std::ranges::sort(chunks, std::less{}, [](auto const& chunk) {
        return chunk->sequence();
    });

    std::vector<PackedData> result;
    result.reserve(chunks.size());
    for (auto& chunk : chunks) {
        result.push_back(chunk->release());
    }
    return result;
}

std::size_t SparseAlltoall::spill(std::optional<std::size_t> amount) {
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE);
        spill_need = headroom < 0 ? safe_cast<std::size_t>(std::abs(headroom)) : 0;
    }

    std::size_t spilled{0};
    // Only spill from the ready (fully received) chunks.
    for (auto& chunks : std::views::values(received_)) {
        if (spilled >= spill_need) {
            break;
        }
        // Ready chunks are at rest, so they may spill all the way to disk.
        spilled += chunks.spill(
            br_, spill_need - spilled, preference_lt(MemoryType::DEVICE, MEMORY_TYPES)
        );
    }
    return spilled;
}

void SparseAlltoall::send_ready_messages() {
    auto const& statistics = comm_->progress_thread()->statistics();
    std::vector<std::unique_ptr<communicator::MetadataPayloadExchange::Message>> messages;
    for (auto& chunk : outgoing_.extract_ready()) {
        if (chunk->data_size() > 0) {
            statistics->add_bytes_stat("sparsealltoall-payload-send", chunk->data_size());
        }
        messages.push_back(
            std::make_unique<communicator::MetadataPayloadExchange::Message>(
                chunk->destination(),
                std::move(*chunk->serialize()),
                chunk->release_data_buffer()
            )
        );
    }
    if (!messages.empty()) {
        exchange_->send(std::move(messages));
    }
}

void SparseAlltoall::receive_messages() {
    auto const& statistics = comm_->progress_thread()->statistics();
    for (auto& msg : exchange_->recv()) {
        auto const src = msg->peer_rank();
        auto chunk =
            detail::Chunk::deserialize(msg->release_metadata(), msg->release_data(), br_);
        RAPIDSMPF_EXPECTS(
            chunk->origin() == src,
            "SparseAlltoall received a chunk with unexpected origin"
        );
        auto it = received_.find(src);
        RAPIDSMPF_EXPECTS(
            it != received_.end(),
            "SparseAlltoall received a message from a rank that is not a source"
        );
        if (chunk->data_size() > 0) {
            statistics->add_bytes_stat("sparsealltoall-payload-recv", chunk->data_size());
        }
        it->second.insert(std::move(chunk));
    }
}

ProgressThread::ProgressState SparseAlltoall::event_loop() {
    send_ready_messages();
    // `insert_finished()` is called after all `insert()` calls, so once it has been
    // called and every inserted chunk has been sent, no more messages will be sent.
    if (locally_finished_.load(std::memory_order_acquire) && outgoing_.empty()) {
        exchange_->finish();
    }
    exchange_->progress();
    receive_messages();

    // After `finish()`, the exchange is idle once all messages from every peer have
    // been received.
    if (exchange_->is_finished() && exchange_->is_idle()) {
        {
            std::lock_guard lock(mutex_);
            can_extract_ = true;
        }
        cv_.notify_all();
        if (auto callback = std::move(finished_callback_)) {
            callback();
        }
        return ProgressThread::ProgressState::Done;
    }
    return ProgressThread::ProgressState::InProgress;
}

}  // namespace rapidsmpf::coll
