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
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
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
    std::shared_ptr<Statistics> statistics,
    std::function<void(void)>&& finished_callback
)
    : comm_{std::move(comm)},
      br_{br},
      statistics_{std::move(statistics)},
      finished_callback_{std::move(finished_callback)},
      finish_counter_{comm_->nranks()},
      op_id_{op_id},
      remote_finish_counter_{comm_->nranks() - 1} {
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
     * 3. for each ready chunk: send metadata to dst and post send for buffer
     * 4. move to chunk to for_extraction_ once send completes
     * 5. chunk is ready for extraction by user
     *
     * Receive side:
     * 1. receive metadata from src
     * 2. allocate chunk and post receive from src
     * 3. Once receive completes
     *  a. If chunk origin is destination (end-of-ring), move to for_extraction_
     *  b. Otherwise insert chunk in inserted_
     *
     * Note: we commit at the point of sending metadata of ready
     * messages to send data immediately. This avoids the need for one
     * ack round-trip.
     */
    Rank const dst = (comm_->rank() + 1) % comm_->nranks();
    Rank const src = (comm_->rank() + comm_->nranks() - 1) % comm_->nranks();
    // GPU data sends and metadata sends can be arbitrarily interleaved. To allow reuse of
    // `op_id` once `wait_and_extract()` returns, we rely on a number of invariants
    // enforced by the communication scheme.
    //
    // Suppose we have two successive allgathers separated by a wait_and_extract "barrier"
    // that reuse the op_id:
    //
    // AG1(op_id)
    // AG1.wait_and_extract()
    // AG2(op_id)
    //
    // The requirements for safe reuse of the tag are that:
    // 1. all metadata sends/receives from AG1 are posted before wait_and_extract returns
    // 2. all data sends/receives are posted before wait_and_extract returns
    //
    // There can be arbitrary interleaving of messages (e.g. finish messages and normal
    // metadata messages), and data messages and metadata messages, as long as these two
    // invariants are upheld.
    //
    // The communication scheme in this loop enforces this in the following way.
    // The finish condition requires that:
    // - we have received finish messages from all ranks, defining the final extraction
    //   goalpost;
    // - The extraction postbox has reached a size equal to the advertised goalpost.
    //
    // Posting receives for more metadata is gated on both of these conditions, so we only
    // post exactly the correct number of receives.
    //
    // To ensure that data sends/receives are correctly posted, note that data is only put
    // in the extraction postbox _after_ it has been posted for send, therefore
    // `wait_and_extract()` cannot return until all sends/receives have at least been
    // posted, upholding the required invariants.
    Tag metadata_tag{op_id_, 0};
    Tag gpu_data_tag{op_id_, 1};
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
    } else {
        // Chunks that are ready to send
        for (auto&& chunk : inserted_.extract_ready()) {
            // Tell the destination about them. All messages (data + finish) share
            // metadata_tag so the no-overtaking guarantee on a single (src, tag) pair
            // ensures current-collective messages arrive before any new-collective
            // messages that reuse the same op_id.
            fire_and_forget_.push_back(
                comm_->send(chunk->serialize(), dst, metadata_tag)
            );
            if (chunk->is_finish()) {
                // Finish chunk contains as sequence number the number
                // of insertions from that rank.
                mark_finish(chunk->sequence());
            } else {
                auto buf = chunk->release_data_buffer();
                sent_posted_.emplace_back(std::move(chunk));
                sent_futures_.emplace_back(
                    comm_->send(std::move(buf), dst, gpu_data_tag)
                );
            }
        }
        // Receive metadata messages. All messages (data + finish) share metadata_tag, so
        // the no-overtaking guarantee ensures current-collective messages arrive before
        // any new-collective messages that reuse the same op_id. While either of these
        // conditions are true, this allgather needs to consume more metadata messages.
        while (remote_finish_counter_ > 0
               || num_received_messages_ < num_expected_messages_)
        {
            auto const msg = comm_->recv_from(src, metadata_tag);
            if (!msg) {
                break;
            }
            auto chunk = detail::Chunk::deserialize(*msg, br_);
            if (chunk->is_finish()) {
                remote_finish_counter_--;
                num_expected_messages_ += chunk->sequence();
                if (chunk->origin() != dst) {
                    fire_and_forget_.push_back(
                        comm_->send(chunk->serialize(), dst, metadata_tag)
                    );
                }
                mark_finish(chunk->sequence());
            } else {
                num_received_messages_++;
                to_receive_.emplace_back(std::move(chunk));
            }
        }
        // Post receives if the chunk is ready
        for (auto&& chunk : to_receive_) {
            if (!chunk->is_ready()) {
                break;
            }
            auto buf = chunk->release_data_buffer();
            receive_posted_.emplace_back(std::move(chunk));
            receive_futures_.emplace_back(comm_->recv(src, gpu_data_tag, std::move(buf)));
        }
        std::erase(to_receive_, nullptr);

        std::ranges::for_each(
            detail::test_some(receive_posted_, receive_futures_, comm_.get()),
            [&](auto&& chunk) {
                if (chunk->origin() == dst) {
                    for_extraction_.insert(std::move(chunk));
                } else {
                    inserted_.insert(std::move(chunk));
                }
            }
        );
        for_extraction_.insert(
            detail::test_some(sent_posted_, sent_futures_, comm_.get())
        );
        if (!fire_and_forget_.empty()) {
            std::ignore = comm_->test_some(fire_and_forget_);
        }
    }
    bool const containers_empty =
        (fire_and_forget_.empty() && sent_posted_.empty() && receive_posted_.empty()
         && sent_futures_.empty() && receive_futures_.empty() && to_receive_.empty()
         && inserted_.empty());
    bool const received_all_data =
        finish_counter_.load(std::memory_order_acquire) == 0
        && extraction_goalpost_.load(std::memory_order_acquire) == for_extraction_.size();
    // Finish progress and notify only if we've received all data and sent all data.
    // The finish counter being zero includes us locally having inserted a finish marker,
    // so it gates whether or not we will insert more into outgoing messages.
    bool const is_done = received_all_data && containers_empty;
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

}  // namespace rapidsmpf::coll
