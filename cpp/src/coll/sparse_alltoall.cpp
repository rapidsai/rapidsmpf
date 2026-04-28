/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

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
      op_id_{op_id},
      finished_callback_{std::move(finished_callback)} {
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be null");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be null");
    auto const size = comm_->nranks();
    auto const self = comm_->rank();
    source_states_.reserve(srcs_.size());
    for (auto src : srcs_) {
        RAPIDSMPF_EXPECTS(
            src >= 0 && src < size && src != self,
            "SparseAlltoall invalid source rank.",
            std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            source_states_.emplace(src, SourceState{}).second,
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
    function_id_ =
        comm_->progress_thread()->add_function([this]() { return event_loop(); });
}

SparseAlltoall::~SparseAlltoall() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        locally_finished_.load(std::memory_order_acquire),
        "Destroying SparseAlltoall without `insert_finished()`"
    );
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
        !locally_finished_.load(std::memory_order_acquire),
        "SparseAlltoall::insert_finished can only be called once"
    );
    for (auto& [dst, ord] : next_ordinal_per_dst_) {
        outgoing_.insert(
            detail::Chunk::from_empty(
                // Number of metadata messages we send to the destination rank, including
                // this finish message.
                ord.load(std::memory_order_acquire) + 1,
                comm_->rank(),
                dst
            )
        );
    }
    locally_finished_.store(true, std::memory_order_release);
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
    auto state_it = source_states_.find(src);
    RAPIDSMPF_EXPECTS(
        state_it != source_states_.end(),
        "SparseAlltoall::extract provided source is not valid"
    );
    {
        std::lock_guard lock{mutex_};
        RAPIDSMPF_EXPECTS(can_extract_, "Extracting before all chunks are ready");
    }
    auto& state = state_it->second;
    std::ranges::sort(state.chunks, std::less{}, [](auto const& chunk) {
        return chunk->sequence();
    });

    std::vector<PackedData> result;
    result.reserve(state.chunks.size());
    for (auto& chunk : state.chunks) {
        result.push_back(chunk->release());
    }
    state.chunks.clear();
    return result;
}

void SparseAlltoall::send_ready_messages() {
    Tag const metadata_tag{op_id_, 0};
    Tag const payload_tag{op_id_, 1};
    for (auto& chunk : outgoing_.extract_ready()) {
        auto const dst = chunk->destination();
        fire_and_forget_.push_back(comm_->send(chunk->serialize(), dst, metadata_tag));
        if (chunk->data_size() > 0) {
            fire_and_forget_.push_back(
                comm_->send(chunk->release_data_buffer(), dst, payload_tag)
            );
        }
    }
}

void SparseAlltoall::receive_metadata_messages() {
    Tag const metadata_tag{op_id_, 0};
    for (auto& [src, state] : source_states_) {
        while (!state.ready()) {
            auto msg = comm_->recv_from(src, metadata_tag);
            if (!msg) {
                break;
            }
            state.received_count++;
            auto chunk = detail::Chunk::deserialize(*msg, br_);
            RAPIDSMPF_EXPECTS(
                chunk->origin() == src,
                "SparseAlltoall received metadata with unexpected origin"
            );
            if (chunk->is_finish()) {
                RAPIDSMPF_EXPECTS(
                    state.expected_count == 0,
                    "SparseAlltoall received duplicate finish control"
                );
                state.expected_count = chunk->sequence();
            } else {
                state.incoming.push_back(std::move(chunk));
            }
        }
    }
}

void SparseAlltoall::receive_data_messages() {
    Tag const payload_tag{op_id_, 1};
    for (auto& [src, state] : source_states_) {
        std::ptrdiff_t processed = 0;
        auto& queue = state.incoming;
        for (auto& chunk : queue) {
            if (!chunk->is_ready()) {
                break;
            }
            processed++;
            if (chunk->data_size() == 0) {
                state.chunks.push_back(std::move(chunk));
            } else {
                auto buffer = chunk->release_data_buffer();
                receive_posted_.push_back(std::move(chunk));
                receive_futures_.push_back(
                    comm_->recv(src, payload_tag, std::move(buffer))
                );
            }
        }
        queue.erase(queue.begin(), queue.begin() + processed);
    }
}

void SparseAlltoall::complete_data_messages() {
    for (auto& chunk : detail::test_some(receive_posted_, receive_futures_, comm_.get()))
    {
        RAPIDSMPF_EXPECTS(
            !chunk->is_finish(), "SparseAlltoall can only complete non-finish chunks"
        );
        auto& state = source_states_[chunk->origin()];
        state.chunks.push_back(std::move(chunk));
    }
}

bool SparseAlltoall::containers_empty() const {
    return outgoing_.empty() && receive_posted_.empty() && receive_futures_.empty()
           && fire_and_forget_.empty()
           && std::ranges::all_of(source_states_, [](auto const& kv) {
                  return kv.second.incoming.empty();
              });
}

ProgressThread::ProgressState SparseAlltoall::event_loop() {
    send_ready_messages();
    receive_metadata_messages();
    receive_data_messages();
    complete_data_messages();
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }

    bool const have_all_data = std::ranges::all_of(source_states_, [&](auto const& src) {
        return src.second.ready();
    });
    if (locally_finished_.load(std::memory_order_acquire) && have_all_data
        && containers_empty())
    {
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
