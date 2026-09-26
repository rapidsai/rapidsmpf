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

#include <cuda/stream>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler {

namespace {
/**
 * @brief Whether the device can *physically* satisfy an allocation of `size` bytes
 * while keeping some headroom for other allocators.
 *
 * Overbooking a reservation only bypasses the BufferResource budget; the bytes
 * still have to come from the GPU. With a fast spill tier (host RAM) spills and
 * restores refill the device faster than the periodic spill drains it, and
 * unbounded overbooking then drives physical usage to 100% until some other
 * allocator (here: the cudf-polars pipeline) hits a real CUDA OOM (jobs
 * 15571/15573/15575). Keep at least `headroom` free.
 */
bool device_has_physical_headroom(std::size_t size) {
    constexpr std::size_t headroom = std::size_t{4} << 30;  // 4 GiB
    auto const [free, total] = rmm::available_device_memory();
    return free > size + headroom;
}
}  // namespace

using namespace detail;

namespace {

/**
 * @brief Convert chunks into messages for communication.
 *
 * This function converts a vector of chunks into messages suitable for sending
 * through the metadata payload exchange. Each chunk is serialized and its data
 * buffer is released to create the message.
 *
 * @param chunks Vector of chunks to convert (will be moved from).
 * @param peer_rank_fn Function to determine the destination rank for each chunk.
 *
 * @return A vector of message unique pointers ready to be sent.
 */
template <typename PeerRankFn>
std::vector<std::unique_ptr<communicator::MetadataPayloadExchange::Message>>
convert_chunks_to_messages(
    std::vector<detail::Chunk>&& chunks, PeerRankFn&& peer_rank_fn
) {
    std::vector<std::unique_ptr<communicator::MetadataPayloadExchange::Message>> messages;
    messages.reserve(chunks.size());

    for (auto&& chunk : chunks) {
        auto dst = peer_rank_fn(chunk);
        auto metadata = std::move(*chunk.serialize());
        auto data = chunk.release_data_buffer();

        messages.push_back(
            std::make_unique<communicator::MetadataPayloadExchange::Message>(
                dst, std::move(metadata), std::move(data)
            )
        );
    }

    return messages;
}

}  // namespace

class Shuffler::Progress {
  public:
    /**
     * @brief Construct a new shuffler progress instance.
     *
     * @param shuffler Reference to the shuffler instance that this will progress.
     */
    Progress(Shuffler& shuffler) : shuffler_(shuffler) {}

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
        auto const& stats = shuffler_.comm_->progress_thread()->statistics();

        // Submit outgoing chunks to the metadata payload exchange
        {
            auto ready_chunks = shuffler_.to_send_.extract_and_restore(
                shuffler_.br_, shuffler_.reservation_memory_types_
            );
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("submit_outgoing", ready_chunks.size());

            if (!ready_chunks.empty()) {
                auto peer_rank_fn = [&shuffler =
                                         shuffler_](detail::Chunk const& chunk) -> Rank {
                    auto dst = shuffler.partition_owner(
                        shuffler.comm_, chunk.part_id(), shuffler.total_num_partitions
                    );
                    shuffler.comm_->logger()->trace(
                        "submitting message to ", dst, ": ", chunk
                    );
                    RAPIDSMPF_EXPECTS(
                        dst != shuffler.comm_->rank(), "sending message to ourselves"
                    );
                    return dst;
                };

                for (auto const& chunk : ready_chunks) {
                    if (chunk.is_data_buffer_set() && chunk.data_size() > 0) {
                        stats->add_bytes_stat("shuffle-payload-send", chunk.data_size());
                    }
                }

                auto messages =
                    convert_chunks_to_messages(std::move(ready_chunks), peer_rank_fn);

                shuffler_.mpe_->send(std::move(messages));
            }
        }

        // Process all communication operations and get completed chunks
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("process_communication");

            shuffler_.mpe_->progress();
            auto completed_messages = shuffler_.mpe_->recv();

            for (auto&& message : completed_messages) {
                auto chunk = detail::Chunk::deserialize(
                    message->metadata(), shuffler_.br_, false, message->release_data()
                );

                RAPIDSMPF_EXPECTS(
                    shuffler_.partition_owner(
                        shuffler_.comm_, chunk.part_id(), shuffler_.total_num_partitions
                    ) == shuffler_.comm_->rank(),
                    "receiving chunk not owned by us"
                );

                if (chunk.data_size() > 0) {
                    stats->add_bytes_stat("shuffle-payload-recv", chunk.data_size());
                }

                shuffler_.insert_into_received(std::move(chunk));
            }
        }

        // Signal the MPE that no more messages will be sent once all application
        // messages have been flushed from to_send_ into the MPE.
        if (!mpe_finish_called_
            && shuffler_.locally_finished_.load(std::memory_order_acquire)
            && shuffler_.to_send_.empty())
        {
            shuffler_.mpe_->finish();
            mpe_finish_called_ = true;
        }

        // There are no messages to be posted, or waiting to be completed.
        bool const containers_empty =
            shuffler_.mpe_->is_idle() && shuffler_.to_send_.empty();
        // We've inserted a finish message and we've received everything we expect.
        bool const is_finished =
            shuffler_.locally_finished_.load(std::memory_order_acquire)
            && shuffler_.finish_counter_.all_finished();
        // Finished and shuffler is no longer active.
        bool const is_done = !shuffler_.active_.load(std::memory_order_acquire)
                             && is_finished && containers_empty;
        // Signal can_extract_ when all chunks have been received and all internal
        // containers are drained. If we own no partitions we "can-extract"
        // immediately, but we only wake a waiter once we've drained internal
        // containers so that we can reuse the op_id for a subsequent shuffle.
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
    bool mpe_finish_called_{false};

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
    PartitionOwner partition_owner_fn,
    std::unique_ptr<communicator::MetadataPayloadExchange> mpe,
    std::vector<MemoryType> spillable_memory_types,
    std::vector<MemoryType> reservation_memory_types
)
    : total_num_partitions{total_num_partitions},
      partition_owner{std::move(partition_owner_fn)},
      br_{br},
      spillable_memory_types_{std::move(spillable_memory_types)},
      reservation_memory_types_{std::move(reservation_memory_types)},
      to_send_{},
      received_{safe_cast<std::size_t>(total_num_partitions)},
      comm_{std::move(comm)},
      mpe_{
          mpe ? std::move(mpe)
              : std::make_unique<communicator::TagMetadataPayloadExchange>(
                    comm_,
                    op_id,
                    [this](std::size_t size) -> std::unique_ptr<Buffer> {
                        auto reservation =
                            br_->try_reserve_or_spill(size, reservation_memory_types_);
                        if (!reservation.has_value()) {
                            // Spilling could not make room: the shuffler has nothing
                            // device-resident left and the budget is held by other users
                            // of the buffer resource (e.g. a streaming pipeline waiting
                            // for this shuffle to advance). Failing here live-locks, so
                            // overbook on the preferred memory type instead, as
                            // rapidsmpf's streaming reservations do; the periodic spill
                            // brings usage back under the limit afterwards.
                            if (reservation_memory_types_.front() == MemoryType::DEVICE
                                && !device_has_physical_headroom(size))
                            {
                                // Budget says no and the GPU really is full: wait for
                                // the periodic spill rather than overbook into an OOM.
                                br_->statistics()->add_bytes_stat(
                                    "recv-overbook-deferred-bytes", size
                                );
                                return nullptr;
                            }
                            auto [overbooked, amount] = br_->reserve(
                                reservation_memory_types_.front(),
                                size,
                                AllowOverbooking::YES
                            );
                            if (overbooked.size() < size) {
                                return nullptr;
                            }
                            br_->statistics()->add_bytes_stat(
                                "recv-overbooked-bytes", amount
                            );
                            reservation = std::move(overbooked);
                        }
                        std::unique_ptr<Buffer> data;
                        try {
                            data = br_->make_buffer(
                                br_->stream_pool()->get_stream(), std::move(*reservation)
                            );
                        } catch (rmm::out_of_memory const&) {
                            // The reservation (possibly overbooked, see above) is only
                            // a promise against the budget; the device can still be
                            // physically full when spills and restores run faster than
                            // the periodic spill (seen with a host spill tier: job
                            // 15571). Letting the exception escape kills the progress
                            // thread and, with it, every peer's connection. Report
                            // "no buffer yet" instead: the exchange retries on the next
                            // progress iteration (bounded by the allocation retry
                            // limit) while spilling frees device memory.
                            br_->statistics()->add_bytes_stat("recv-alloc-oom-retry-bytes", size);
                            return nullptr;
                        }
                        if (data->mem_type() == MemoryType::PINNED_HOST
                            || data->mem_type() == MemoryType::HOST)
                        {
                            br_->statistics()->add_bytes_stat(
                                "recv-into-host-memory", size
                            );
                        }
                        return data;
                    },
                    comm_->progress_thread()->statistics()
                )
      },
      local_partitions_{local_partitions(comm_, total_num_partitions, partition_owner)},
      finish_counter_{comm_->nranks(), safe_cast<PartID>(local_partitions_.size())},
      outbound_chunk_counter_(safe_cast<std::size_t>(comm_->nranks()), 0),
      finished_callback_{std::move(finished_callback)} {
    RAPIDSMPF_EXPECTS(
        total_num_partitions > 0, "number of partitions must be strictly positive"
    );
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(
        !reservation_memory_types_.empty(),
        "reservation_memory_types cannot be empty",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        std::ranges::all_of(
            reservation_memory_types_,
            [](auto mem_type) { return contains(ADDRESSABLE_MEMORY_TYPES, mem_type); }
        ),
        "reservation_memory_types contains a non-addressable memory type",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        std::ranges::all_of(
            spillable_memory_types_,
            [](auto mem_type) {
                return mem_type != MemoryType::DEVICE && contains(MEMORY_TYPES, mem_type);
            }
        ),
        "spillable_memory_types contains an invalid spill destination",
        std::invalid_argument
    );

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
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE);
        if (headroom < 0 && packed_data.data
            && packed_data.data->mem_type() == MemoryType::DEVICE)
        {
            auto chunk = create_chunk(pid, std::move(packed_data));
            auto reservation =
                br_->reserve_or_fail(chunk.data_size(), spillable_memory_types_);
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
    if (spillable_memory_types_.empty()) {
        return 0;
    }
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE);
        if (headroom < 0) {
            spill_need = safe_cast<std::size_t>(std::abs(headroom));
        }
    }
    std::size_t spilled{0};
    if (spill_need > 0) {
        spilled = received_.spill(br_, spill_need, spillable_memory_types_);
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
