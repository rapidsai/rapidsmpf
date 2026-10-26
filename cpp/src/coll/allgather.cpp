/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::coll {

void AllGather::insert(std::uint64_t sequence_number, PackedData&& packed_data) {
    nlocal_insertions_.fetch_add(1, std::memory_order_relaxed);
    return insert(
        detail::Chunk::from_packed_data(
            sequence_number,
            comm_->rank(),
            detail::Chunk::INVALID_RANK,  // Destination is hard-coded in ring algorithm
            std::move(packed_data)
        )

    );
}

void AllGather::insert(std::unique_ptr<detail::Chunk> chunk) {
    RAPIDSMPF_EXPECTS(
        !locally_finished_.load(std::memory_order_acquire),
        "Can't insert after locally indicating finished"
    );
    inserted_.insert(std::move(chunk));
}

void AllGather::insert_finished() {
    inserted_.insert(
        detail::Chunk::from_empty(
            nlocal_insertions_.load(std::memory_order_acquire),
            comm_->rank(),
            detail::Chunk::INVALID_RANK
        )
    );
    locally_finished_.store(true, std::memory_order_release);
}

void AllGather::mark_finish(std::uint64_t expected_chunks) noexcept {
    // We must increment the extraction goalpost before decrementing the finish
    // counter so that we cannot, on another thread, observe a finish counter of zero
    // with chunks still to be received.
    extraction_goalpost_.fetch_add(expected_chunks, std::memory_order_acq_rel);
    finish_counter_.fetch_sub(1, std::memory_order_relaxed);
}

std::vector<PackedData> AllGather::wait_and_extract(
    AllGather::Ordered ordered, std::chrono::milliseconds timeout
) {
    wait(timeout);
    auto chunks = for_extraction_.extract();
    extraction_goalpost_.fetch_sub(chunks.size(), std::memory_order_acq_rel);
    std::vector<PackedData> result;
    result.reserve(chunks.size());
    if (ordered == AllGather::Ordered::YES) {
        std::ranges::sort(chunks, std::less{}, [](auto&& chunk) { return chunk->id(); });
    }
    std::ranges::transform(chunks, std::back_inserter(result), [](auto&& chunk) {
        return chunk->release();
    });
    return result;
}

void AllGather::wait(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (timeout < std::chrono::milliseconds{0}) {
        cv_.wait(lock, [&]() { return can_extract_; });
    } else {
        RAPIDSMPF_EXPECTS(
            cv_.wait_for(lock, timeout, [&]() { return can_extract_; }),
            "wait timeout reached",
            std::runtime_error
        );
    }
}

std::size_t AllGather::spill(std::optional<std::size_t> amount) {
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE);
        spill_need = headroom < 0 ? safe_cast<std::size_t>(std::abs(headroom)) : 0;
    }
    std::size_t spilled{0};
    if (spill_need > 0) {
        // Spill from ready post box then inserted postbox
        spilled = for_extraction_.spill(br_, spill_need);
        if (spilled < spill_need) {
            spilled += inserted_.spill(br_, spill_need - spilled);
        }
    }
    return spilled;
}

AllGather::~AllGather() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        locally_finished_.load(std::memory_order_acquire),
        "Destroying allgather without `insert_finished()`"
    );
    br_->spill_manager().remove_spill_function(spill_function_id_);
    comm_->progress_thread()->remove_function(function_id_);
}

AllGather::AllGather(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    BufferResource* br,
    std::function<void(void)>&& finished_callback
)
    : comm_{std::move(comm)},
      br_{br},
      finished_callback_{std::move(finished_callback)},
      finish_counter_{comm_->nranks()} {
    // The ring neighbours, which are the same rank with two ranks.
    std::vector<Rank> peers;
    if (comm_->nranks() > 1) {
        peers.push_back((comm_->rank() + 1) % comm_->nranks());
        Rank const src = (comm_->rank() + comm_->nranks() - 1) % comm_->nranks();
        if (src != peers.front()) {
            peers.push_back(src);
        }
    }
    // Received chunks are forwarded, so forced receives are not moved to disk.
    exchange_ = std::make_unique<communicator::TagMetadataPayloadExchange>(
        comm_,
        op_id,
        std::move(peers),
        communicator::default_allocate_buffer,
        br_->memory_reserve_timeout(),
        br_,
        false
    );
    function_id_ =
        comm_->progress_thread()->add_function([this]() { return event_loop(); });
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );
}

ProgressThread::ProgressState AllGather::event_loop() {
    RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("AllGather::event_loop");
    /*
     * Data flow:
     * User inserts into inserted_
     * Send side:
     * 1. chunk inserted
     * 2. extract ready chunks
     * 3. for each ready chunk: send it to dst, keeping it in sent_posted_
     * 4. once the exchange returns its data, move the chunk to for_extraction_
     * 5. chunk is ready for extraction by user
     *
     * Receive side:
     * 1. receive chunk from src
     * 2. if chunk origin is destination (end-of-ring), move to for_extraction_
     * 3. otherwise insert chunk in inserted_ to forward it
     *
     * Finish chunks carry the number of insertions of their origin and are forwarded
     * around the ring as soon as they are received.
     *
     * Once this rank will send nothing more, it calls `finish()` on the exchange. The
     * exchange is then idle once all messages from both neighbours have been received,
     * which makes it safe to reuse `op_id` after `wait_and_extract()` returns.
     */
    Rank const dst = (comm_->rank() + 1) % comm_->nranks();
    auto const& statistics = comm_->progress_thread()->statistics();
    bool is_done{false};
    if (comm_->nranks() == 1) {
        // Note that we don't need to use extract_ready because there is no message
        // passing and our promise to the consumer is that extracted data chunks are valid
        // on their respective streams.
        for (auto&& chunk : inserted_.extract()) {
            if (chunk->is_finish()) {
                mark_finish(chunk->sequence());
            } else {
                for_extraction_.insert(std::move(chunk));
            }
        }
        is_done = received_all_data();
    } else {
        using Message = communicator::MetadataPayloadExchange::Message;
        // Send the chunks that are ready.
        std::vector<std::unique_ptr<Message>> messages;
        for (auto&& chunk : inserted_.extract_ready()) {
            if (chunk->is_finish()) {
                // Finish chunk contains as sequence number the number
                // of insertions from that rank.
                messages.push_back(
                    std::make_unique<Message>(dst, std::move(*chunk->serialize()))
                );
                mark_finish(chunk->sequence());
            } else {
                if (chunk->data_size() > 0) {
                    statistics->add_bytes_stat(
                        "allgather-payload-send", chunk->data_size()
                    );
                }
                messages.push_back(
                    std::make_unique<Message>(
                        dst,
                        std::move(*chunk->serialize()),
                        chunk->release_data_buffer(),
                        communicator::MetadataPayloadExchange::ReturnPayload::YES
                    )
                );
                sent_posted_.push_back(std::move(chunk));
            }
        }
        if (!messages.empty()) {
            exchange_->send(std::move(messages));
        }
        exchange_->progress();

        // Sent chunks get their data back in send order.
        for (auto&& msg : exchange_->release_sent()) {
            RAPIDSMPF_EXPECTS(!sent_posted_.empty(), "unexpected sent message");
            auto chunk = std::move(sent_posted_.front());
            sent_posted_.pop_front();
            chunk->attach_data_buffer(msg->release_data());
            for_extraction_.insert(std::move(chunk));
        }

        for (auto&& msg : exchange_->recv()) {
            auto chunk =
                detail::Chunk::deserialize(msg->metadata(), msg->release_data(), br_);
            if (chunk->is_finish()) {
                num_expected_messages_ += chunk->sequence();
                if (chunk->origin() != dst) {
                    exchange_->send(
                        std::make_unique<Message>(dst, msg->release_metadata())
                    );
                }
                mark_finish(chunk->sequence());
            } else {
                num_received_messages_++;
                if (chunk->data_size() > 0) {
                    statistics->add_bytes_stat(
                        "allgather-payload-recv", chunk->data_size()
                    );
                }
                if (chunk->origin() == dst) {
                    for_extraction_.insert(std::move(chunk));
                } else {
                    inserted_.insert(std::move(chunk));
                }
            }
        }

        // Once the finish chunks of all ranks are known, all chunks have been received
        // and every chunk to forward has been sent, this rank sends nothing more.
        if (!exchange_finished_ && finish_counter_.load(std::memory_order_acquire) == 0
            && num_received_messages_ == num_expected_messages_ && inserted_.empty())
        {
            exchange_->finish();
            exchange_finished_ = true;
        }
        is_done = received_all_data() && exchange_finished_ && exchange_->is_idle()
                  && sent_posted_.empty();
    }
    if (is_done) {
        // We can release our output buffers so notify a waiter.
        {
            std::lock_guard lock(mutex_);
            can_extract_ = true;
        }
        cv_.notify_one();
        if (auto callback = std::move(finished_callback_)) {
            callback();
        }
        return ProgressThread::ProgressState::Done;
    }
    return ProgressThread::ProgressState::InProgress;
}

bool AllGather::received_all_data() const noexcept {
    // The finish counter being zero includes us locally having inserted a finish marker,
    // so it gates whether or not we will insert more into outgoing messages.
    return inserted_.empty() && finish_counter_.load(std::memory_order_acquire) == 0
           && extraction_goalpost_.load(std::memory_order_acquire)
                  == for_extraction_.size();
}

}  // namespace rapidsmpf::coll
