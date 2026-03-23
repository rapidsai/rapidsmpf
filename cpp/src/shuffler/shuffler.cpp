/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler {

using namespace detail;

class Shuffler::Progress {
  public:
    /**
     * @brief Construct a new shuffler progress instance.
     *
     * @param shuffler Reference to the shuffler instance that this will progress.
     */
    Progress(Shuffler& shuffler)
        : shuffler_(shuffler),
          peer_received_(safe_cast<std::size_t>(shuffler.comm_->nranks()), 0),
          peer_expected_(safe_cast<std::size_t>(shuffler.comm_->nranks()), 0) {}

    /**
     * @brief Executes a single iteration of the shuffler's event loop.
     *
     * This function manages the movement of data chunks between ranks in the distributed
     * system, handling tasks such as sending and receiving metadata and GPU data. It also
     * manages the processing of chunks in transit, both outgoing and incoming, and
     * updates the necessary data structures for further processing.
     *
     * @return The progress state of the shuffler.
     */
    ProgressThread::ProgressState operator()() {
        RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("Shuffler.Progress", p_iters++);
        auto const t0_event_loop = Clock::now();

        // Tags for each stage of the shuffle
        Tag const metadata_tag{shuffler_.op_id_, 0};
        Tag const gpu_data_tag{shuffler_.op_id_, 1};

        auto& log = *shuffler_.comm_->logger();
        auto& stats = *shuffler_.statistics_;

        {
            auto const t0_send = Clock::now();
            auto chunks = shuffler_.to_send_.extract_ready();
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("shuffle_send", chunks.size());
            for (auto&& chunk : chunks) {
                auto dst = shuffler_.partition_owner(
                    shuffler_.comm_, chunk.part_id(), shuffler_.total_num_partitions
                );
                log.trace("send to ", dst, ": ", chunk);
                RAPIDSMPF_EXPECTS(
                    dst != shuffler_.comm_->rank(), "sending chunk to ourselves"
                );

                fire_and_forget_.push_back(
                    shuffler_.comm_->send(chunk.serialize(), dst, metadata_tag)
                );
                if (chunk.data_size() > 0) {
                    shuffler_.statistics_->add_bytes_stat(
                        "shuffle-payload-send", chunk.data_size()
                    );
                    fire_and_forget_.push_back(shuffler_.comm_->send(
                        chunk.release_data_buffer(), dst, gpu_data_tag
                    ));
                }
            }
            stats.add_duration_stat("event-loop-send", Clock::now() - t0_send);
        }

        // Receive incoming metadata of remote chunks and place them in
        // `incoming_chunks_`, using per-peer recv_from to avoid consuming
        // messages belonging to a future collective on the same tag.
        {
            auto const t0_metadata_recv = Clock::now();
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("meta_recv");
            [[maybe_unused]] int recv_from_iters =
                0;  // this will be stripped off if RAPIDSMPF_VERBOSE_INFO is not set
            if (!shuffler_.local_partitions_.empty()) {
                for (Rank peer = 0; peer < shuffler_.comm_->nranks(); ++peer) {
                    if (peer == shuffler_.comm_->rank()) {
                        continue;
                    }
                    auto const p = safe_cast<std::size_t>(peer);
                    while (peer_expected_[p] == 0
                           || peer_received_[p] < peer_expected_[p])
                    {
                        auto msg = shuffler_.comm_->recv_from(peer, metadata_tag);
                        if (!msg) {
                            break;
                        }
                        auto chunk =
                            Chunk::deserialize(*msg, shuffler_.br_, /*validate=*/false);
                        log.trace("recv_from ", peer, ": ", chunk);
                        peer_received_[p]++;
                        if (chunk.is_control_message()) {
                            peer_expected_[p] = chunk.expected_num_chunks();
                        } else {
                            RAPIDSMPF_EXPECTS(
                                shuffler_.partition_owner(
                                    shuffler_.comm_,
                                    chunk.part_id(),
                                    shuffler_.total_num_partitions
                                ) == shuffler_.comm_->rank(),
                                "receiving chunk not owned by us"
                            );
                        }
                        incoming_chunks_[peer].push_back(std::move(chunk));
                        recv_from_iters++;
                    }
                }
            }
            stats.add_duration_stat(
                "event-loop-metadata-recv", Clock::now() - t0_metadata_recv
            );
            RAPIDSMPF_NVTX_MARKER_VERBOSE("meta_recv_iters", recv_from_iters);
        }

        // Post receives for incoming chunks. Note that we start the allocation of chunks
        // in received message order, but because the allocations run on different streams
        // they might not complete and be ready in that order. To handle that, we separate
        // incoming chunks by rank and then process chunks in FIFO order until we observe
        // a non-ready chunk.
        {
            auto const t0_post_incoming_chunk_recv = Clock::now();
            for (auto& [src, chunks] : incoming_chunks_) {
                RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("post_chunk_recv", chunks.size());
                std::ptrdiff_t n_processed = 0;
                for (auto& chunk : chunks) {
                    log.trace("checking incoming chunk data from ", src, ": ", chunk);

                    if (chunk.data_size() > 0) {
                        if (!chunk.is_ready()) {
                            break;
                        }

                        auto chunk_id = chunk.chunk_id();
                        auto data_size = chunk.data_size();

                        // Setup to receive the chunk into `in_transit_*`.
                        auto future = shuffler_.comm_->recv(
                            src, gpu_data_tag, chunk.release_data_buffer()
                        );
                        RAPIDSMPF_EXPECTS(
                            in_transit_futures_.emplace(chunk_id, std::move(future))
                                .second,
                            "in transit future already exist"
                        );
                        RAPIDSMPF_EXPECTS(
                            in_transit_chunks_.emplace(chunk_id, std::move(chunk)).second,
                            "in transit chunk already exist"
                        );
                        shuffler_.statistics_->add_bytes_stat(
                            "shuffle-payload-recv", data_size
                        );
                    } else {
                        // Control messages and metadata-only messages go
                        // directly to the ready postbox.
                        shuffler_.insert_into_received(std::move(chunk));
                    }
                    n_processed++;
                }
                chunks.erase(chunks.begin(), chunks.begin() + n_processed);
            }

            stats.add_duration_stat(
                "event-loop-post-incoming-chunk-recv",
                Clock::now() - t0_post_incoming_chunk_recv
            );
        }

        // Check if any data in transit is finished.
        {
            auto const t0_check_future_finish = Clock::now();
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE(
                "check_fut_finish", in_transit_futures_.size()
            );
            if (!in_transit_futures_.empty()) {
                std::vector<ChunkID> finished =
                    shuffler_.comm_->test_some(in_transit_futures_);
                for (auto cid : finished) {
                    auto chunk = extract_value(in_transit_chunks_, cid);
                    auto future = extract_value(in_transit_futures_, cid);
                    chunk.set_data_buffer(
                        shuffler_.comm_->release_data(std::move(future))
                    );

                    shuffler_.insert_into_received(std::move(chunk));
                }
            }

            // Check if we can free some of the outstanding futures.
            if (!fire_and_forget_.empty()) {
                std::ignore = shuffler_.comm_->test_some(fire_and_forget_);
            }
            stats.add_duration_stat(
                "event-loop-check-future-finish", Clock::now() - t0_check_future_finish
            );
        }

        stats.add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);

        // There are no messages to be posted, or waiting to be completed.
        bool const containers_empty =
            fire_and_forget_.empty()
            && std::ranges::all_of(
                incoming_chunks_, [](auto const& kv) { return kv.second.empty(); }
            )
            && in_transit_chunks_.empty() && in_transit_futures_.empty()
            && shuffler_.to_send_.empty();
        // We've inserted a finish message and we've received everything we expect.
        bool const is_finished =
            shuffler_.locally_finished_.load(std::memory_order_acquire)
            && shuffler_.finish_counter_.all_finished();
        // Finished and shuffler is no longer active.
        bool const is_done = !shuffler_.active_.load(std::memory_order_acquire)
                             && is_finished && containers_empty;
        // Signal can_extract_ when all chunks have been received and all internal
        // containers are drained. If we own no partitions we "can-extract" immediately,
        // but we only wake a waiter once we've drained internal containers so that we can
        // reuse the op_id for a subsequent shuffle.
        if (!shuffler_.can_extract_ && is_finished && containers_empty) {
            {
                std::lock_guard lock(shuffler_.mutex_);
                shuffler_.can_extract_ = true;
            }
            shuffler_.cv_.notify_all();
            if (auto callback = std::move(shuffler_.finished_callback_)) {
                callback();
            }
        }
        return is_done ? ProgressThread::ProgressState::Done
                       : ProgressThread::ProgressState::InProgress;
    }

  private:
    Shuffler& shuffler_;
    std::vector<std::size_t> peer_received_;  ///< Messages received per rank.
    std::vector<std::size_t> peer_expected_;  ///< Total expected from rank (0 = unknown).
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::unordered_map<Rank, std::vector<detail::Chunk>>
        incoming_chunks_;  ///< Per-rank FIFO of chunks awaiting receive.
    std::unordered_map<detail::ChunkID, detail::Chunk>
        in_transit_chunks_;  ///< Chunks currently in transit.
    std::unordered_map<detail::ChunkID, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit chunks.

#if RAPIDSMPF_VERBOSE_INFO
    std::int64_t p_iters = 0;  ///< Number of progress iterations (for NVTX)
#endif
};

std::vector<PartID> Shuffler::local_partitions(
    std::shared_ptr<Communicator> const& comm,
    PartID total_num_partitions,
    PartitionOwner partition_owner
) {
    std::vector<PartID> ret;
    for (PartID i = 0; i < total_num_partitions; ++i) {
        if (partition_owner(comm, i, total_num_partitions) == comm->rank()) {
            ret.push_back(i);
        }
    }
    return ret;
}

Shuffler::Shuffler(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    PartID total_num_partitions,
    BufferResource* br,
    FinishedCallback&& finished_callback,
    PartitionOwner partition_owner_fn
)
    : total_num_partitions{total_num_partitions},
      partition_owner{std::move(partition_owner_fn)},
      br_{br},
      op_id_{op_id},
      to_send_{},
      received_{safe_cast<std::size_t>(total_num_partitions)},
      comm_{std::move(comm)},
      local_partitions_{local_partitions(comm_, total_num_partitions, partition_owner)},
      finish_counter_{comm_->nranks(), safe_cast<PartID>(local_partitions_.size())},
      outbound_chunk_counter_(safe_cast<std::size_t>(comm_->nranks()), 0),
      statistics_{br_->statistics()},
      finished_callback_{std::move(finished_callback)} {
    RAPIDSMPF_EXPECTS(
        total_num_partitions > 0, "number of partitions must be strictly positive"
    );
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be NULL");

    // We need to register the progress function with the progress thread, but
    // that cannot be done in the constructor's initializer list because the
    // Shuffler isn't fully constructed yet.
    // NB: this only works because `Shuffler` is not movable, otherwise if moved,
    // `this` will become invalid.
    progress_thread_function_id_ = comm_->progress_thread()->add_function(
        [progress = std::make_shared<Progress>(*this)]() { return (*progress)(); }
    );

    // Register a spill function that spill buffers in this shuffler.
    // Note, the spill function can use `this` because a Shuffler isn't movable.
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );
}

std::span<PartID const> Shuffler::local_partitions() const {
    return local_partitions_;
}

Shuffler::~Shuffler() {
    shutdown();
}

void Shuffler::shutdown() {
    RAPIDSMPF_EXPECTS_FATAL(
        locally_finished_.load(std::memory_order_acquire),
        "Destroying shuffler without `insert_finished()`"
    );
    bool expected = true;
    if (active_.compare_exchange_strong(expected, false)) {
        auto& log = comm_->logger();
        log->debug("Shuffler.shutdown() - initiate");
        comm_->progress_thread()->remove_function(progress_thread_function_id_);
        br_->spill_manager().remove_spill_function(spill_function_id_);
        log->debug("Shuffler.shutdown() - done");
    }
}

detail::Chunk Shuffler::create_chunk(PartID pid, PackedData&& packed_data) {
    return detail::Chunk::from_packed_data(get_new_cid(), pid, std::move(packed_data));
}

void Shuffler::insert_into_received(detail::Chunk&& chunk) {
    auto& log = comm_->logger();
    log->trace("insert_into_received: ", chunk);

    if (chunk.is_control_message()) {
        Rank src_rank = extract_rank(chunk.chunk_id());
        finish_counter_.move_goalpost(src_rank, chunk.expected_num_chunks());
    } else {
        received_.insert(std::move(chunk));
    }
    finish_counter_.add_finished_chunk();
}

void Shuffler::insert(detail::Chunk&& chunk) {
    Rank dst_rank = partition_owner(comm_, chunk.part_id(), total_num_partitions);
    if (!chunk.is_control_message()) {
        std::atomic_ref(outbound_chunk_counter_[safe_cast<std::size_t>(dst_rank)])
            .fetch_add(1, std::memory_order_relaxed);
    }
    if (dst_rank == comm_->rank()) {
        // this is a local chunk, so we can move it straight to received.
        insert_into_received(std::move(chunk));
    } else {
        // this is a remote chunk, so we need to send it
        to_send_.insert(std::make_unique<detail::Chunk>(std::move(chunk)));
    }
}

void Shuffler::insert(std::unordered_map<PartID, PackedData>&& chunks) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_EXPECTS(
        !locally_finished_.load(std::memory_order_acquire),
        "Can't insert after locally indicating finished"
    );
    // Insert each chunk into the inbox.
    for (auto& [pid, packed_data] : chunks) {
        if (packed_data.empty()) {  // skip empty packed data
            continue;
        }

        // Check if we should spill the chunk before inserting into the inbox.
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        if (headroom < 0 && packed_data.data) {
            auto reservation =
                br_->reserve_or_fail(packed_data.data->size, SPILL_TARGET_MEMORY_TYPES);
            auto chunk = create_chunk(pid, std::move(packed_data));
            // Spill the new chunk before inserting.
            chunk.set_data_buffer(br_->move(chunk.release_data_buffer(), reservation));
            insert(std::move(chunk));
        } else {
            insert(create_chunk(pid, std::move(packed_data)));
        }
    }

    // Spill if current available device memory is still negative.
    br_->spill_manager().spill_to_make_headroom(0);
}

void Shuffler::insert_finished() {
    // All insert() calls happen-before this point (API contract), so the
    // relaxed atomic_ref increments are visible and we can read directly.
    auto const& counts = outbound_chunk_counter_;

    // Pick an arbitrary representative partition for each rank so we can route one
    // control message per rank. Ranks that own no partitions do not need to be sent a
    // control message because they will never receive any messages.
    std::vector<PartID> representative_pid(
        safe_cast<std::size_t>(comm_->nranks()), total_num_partitions
    );
    for (PartID p = 0; p < total_num_partitions; ++p) {
        auto r = safe_cast<std::size_t>(partition_owner(comm_, p, total_num_partitions));
        representative_pid[r] = p;
    }

    for (std::size_t r = 0; r < counts.size(); ++r) {
        auto pid = representative_pid[r];
        if (pid == total_num_partitions) {
            continue;  // rank owns no partitions
        }
        insert(detail::Chunk::from_finished_partition(get_new_cid(), pid, counts[r] + 1));
    }
    locally_finished_.store(true, std::memory_order_release);
}

std::vector<PackedData> Shuffler::extract(PartID pid) {
    RAPIDSMPF_NVTX_FUNC_RANGE();

    // Quick return if the partition is empty.
    if (received_.is_empty(pid)) {
        return std::vector<PackedData>{};
    }

    auto chunks = received_.extract(pid);

    std::vector<PackedData> ret;
    ret.reserve(chunks.size());

    std::ranges::transform(
        chunks, std::back_inserter(ret), [](auto&& chunk) -> PackedData {
            return {chunk.release_metadata_buffer(), chunk.release_data_buffer()};
        }
    );

    return ret;
}

bool Shuffler::finished() const {
    return finish_counter_.all_finished() && received_.empty();
}

void Shuffler::wait(std::optional<std::chrono::milliseconds> timeout) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::unique_lock lock(mutex_);
    if (timeout.has_value()) {
        RAPIDSMPF_EXPECTS(
            cv_.wait_for(lock, *timeout, [&] { return can_extract_; }),
            "wait timeout reached",
            std::runtime_error
        );
    } else {
        cv_.wait(lock, [&] { return can_extract_; });
    }
}

std::size_t Shuffler::spill(std::optional<std::size_t> amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        if (headroom < 0) {
            spill_need = safe_cast<std::size_t>(std::abs(headroom));
        }
    }
    std::size_t spilled{0};
    if (spill_need > 0) {
        spilled = received_.spill(br_, spill_need);
    }
    return spilled;
}

detail::ChunkID Shuffler::get_new_cid() {
    // Place the counter in the last 38 bits (supports 256G chunks).
    std::uint64_t lower = ++chunk_id_counter_;
    // and place the rank in the first 26 bits (supports 64M ranks).
    auto upper = safe_cast<std::uint64_t>(comm_->rank()) << chunk_id_counter_bits;
    return upper | lower;
}

std::string Shuffler::str() const {
    std::stringstream ss;
    ss << "Shuffler(to_send=" << to_send_ << ", received=" << received_ << ", "
       << finish_counter_ << ")";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler
