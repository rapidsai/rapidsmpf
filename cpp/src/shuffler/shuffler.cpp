/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

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
        auto metadata = *chunk.serialize();
        auto data = chunk.release_data_buffer();

        messages.push_back(
            std::make_unique<communicator::MetadataPayloadExchange::Message>(
                dst, std::move(metadata), std::move(data)
            )
        );
    }

    return messages;
}

/**
 * @brief Spills memory buffers within a postbox, e.g., from device to host memory.
 *
 * This function moves a specified amount of memory from device to host storage
 * or another lower-priority memory space, helping manage limited GPU memory
 * by offloading excess data.
 *
 * The spilling is stream-ordered on the individual CUDA stream of each spilled buffer.
 *
 * @note While spilling, chunks are temporarily extracted from the postbox thus other
 * threads trying to extract a chunk that is in the process of being spilled, will fail.
 * To avoid this, the Shuffler uses `outbox_spillling_mutex_` to serialize extractions.
 *
 * @param br Buffer resource for GPU data allocations.
 * @param amount The maximum amount of data (in bytes) to be spilled.
 *
 * @return The actual amount of data successfully spilled from the postbox.
 *
 * @warning This may temporarily empty the postbox, causing emptiness checks to return
 * true even though the postbox is not actually empty. As a result, in the current
 * implementation `postbox_spilling()` must not be used to spill `outgoing_postbox_`.
 */
template <typename KeyType>
std::size_t postbox_spilling(
    BufferResource* br, PostBox<KeyType>& postbox, std::size_t amount
) {
    RAPIDSMPF_NVTX_FUNC_RANGE(amount);
    // Let's look for chunks to spill in the outbox.
    auto const chunk_info = postbox.search(MemoryType::DEVICE);
    std::size_t total_spilled{0};
    for (auto [pid, cid, size] : chunk_info) {
        if (size == 0) {  // skip empty data buffers
            continue;
        }

        // TODO: Use a clever strategy to decide which chunks to spill. For now, we
        // just spill the chunks in an arbitrary order.
        auto reservation = br->reserve_or_fail(size, SPILL_TARGET_MEMORY_TYPES);
        // We extract the chunk, spilled it, and insert it back into the PostBox.
        auto chunk = postbox.extract(pid, cid);
        chunk.set_data_buffer(br->move(chunk.release_data_buffer(), reservation));
        postbox.insert(std::move(chunk));
        if ((total_spilled += size) >= amount) {
            break;
        }
    }
    RAPIDSMPF_NVTX_MARKER("postbox_spilling::total_spilled", total_spilled);
    return total_spilled;
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
        auto const t0_event_loop = Clock::now();

        auto& stats = *shuffler_.statistics_;

        // Submit outgoing chunks to the metadata payload exchange
        {
            auto const t0_submit_outgoing = Clock::now();
            auto ready_chunks = shuffler_.outgoing_postbox_.extract_all_ready();
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
                    if (chunk.data_size() > 0) {
                        stats.add_bytes_stat("shuffle-payload-send", chunk.data_size());
                    }
                }

                auto messages =
                    convert_chunks_to_messages(std::move(ready_chunks), peer_rank_fn);

                shuffler_.mpe_->send(std::move(messages));
            }
            stats.add_duration_stat(
                "event-loop-submit-outgoing", Clock::now() - t0_submit_outgoing
            );
        }

        // Process all communication operations and get completed chunks
        {
            auto const t0_process_comm = Clock::now();
            RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("process_communication");

            shuffler_.mpe_->progress();
            auto completed_messages = shuffler_.mpe_->recv();

            for (auto&& message : completed_messages) {
                auto chunk =
                    detail::Chunk::deserialize(message->metadata(), shuffler_.br_, false);
                if (message->data() != nullptr) {
                    std::ignore = chunk.release_data_buffer();
                    chunk.set_data_buffer(message->release_data());
                }

                RAPIDSMPF_EXPECTS(
                    shuffler_.partition_owner(
                        shuffler_.comm_, chunk.part_id(), shuffler_.total_num_partitions
                    ) == shuffler_.comm_->rank(),
                    "receiving chunk not owned by us"
                );

                if (chunk.data_size() > 0) {
                    stats.add_bytes_stat("shuffle-payload-recv", chunk.data_size());
                }

                shuffler_.insert_into_ready_postbox(std::move(chunk));
            }

            stats.add_duration_stat(
                "event-loop-process-communication", Clock::now() - t0_process_comm
            );
        }

        stats.add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);

        // Return Done only if the shuffler is inactive (shutdown was called) _and_
        // all containers are empty (all work is done).
        return (shuffler_.active_.load(std::memory_order_acquire)
                || !shuffler_.mpe_->is_idle() || !shuffler_.outgoing_postbox_.empty())
                   ? ProgressThread::ProgressState::InProgress
                   : ProgressThread::ProgressState::Done;
    }

  private:
    Shuffler& shuffler_;

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
    std::unique_ptr<communicator::MetadataPayloadExchange> mpe
)
    : total_num_partitions{total_num_partitions},
      partition_owner{std::move(partition_owner_fn)},
      br_{br},
      outgoing_postbox_{
          [this](PartID pid) -> Rank {
              return this->partition_owner(this->comm_, pid, this->total_num_partitions);
          },  // extract Rank from pid
          safe_cast<std::size_t>(comm->nranks())
      },
      ready_postbox_{
          [](PartID pid) -> PartID { return pid; },  // identity mapping
          safe_cast<std::size_t>(total_num_partitions),
      },
      comm_{std::move(comm)},
      mpe_{
          mpe ? std::move(mpe)
              : std::make_unique<communicator::TagMetadataPayloadExchange>(
                    comm_,
                    op_id,
                    [this](std::size_t size) -> std::unique_ptr<Buffer> {
                        return br_->allocate(
                            br_->stream_pool().get_stream(),
                            br_->reserve_or_fail(size, MEMORY_TYPES)
                        );
                    },
                    br_->statistics()
                )
      },
      op_id_{op_id},
      local_partitions_{local_partitions(comm_, total_num_partitions, partition_owner)},
      finish_counter_{comm_->nranks(), local_partitions_, std::move(finished_callback)},
      statistics_{br_->statistics()} {
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

void Shuffler::insert_into_ready_postbox(detail::Chunk&& chunk) {
    auto& log = comm_->logger();
    log->trace("insert_into_outbox: ", chunk);

    auto pid = chunk.part_id();
    if (chunk.is_control_message()) {
        finish_counter_.move_goalpost(pid, chunk.expected_num_chunks());
    } else {
        ready_postbox_.insert(std::move(chunk));
    }
    finish_counter_.add_finished_chunk(pid);
}

void Shuffler::insert(detail::Chunk&& chunk) {
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        ++outbound_chunk_counter_[chunk.part_id()];
    }

    Rank p0_target_rank = partition_owner(comm_, chunk.part_id(), total_num_partitions);
    if (p0_target_rank == comm_->rank()) {
        // this is a local chunk, so we can insert it into the ready postbox
        if (chunk.is_data_buffer_set()) {
            statistics_->add_bytes_stat("shuffle-payload-send", chunk.data_size());
            statistics_->add_bytes_stat("shuffle-payload-recv", chunk.data_size());
        }
        insert_into_ready_postbox(std::move(chunk));
    } else {
        // this is a remote chunk, so we need to insert it into the outgoing postbox
        outgoing_postbox_.insert(std::move(chunk));
    }
}

void Shuffler::insert(std::unordered_map<PartID, PackedData>&& chunks) {
    RAPIDSMPF_NVTX_FUNC_RANGE();

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

void Shuffler::insert_finished(PartID pid) {
    insert_finished(std::vector<PartID>{pid});
}

void Shuffler::insert_finished(std::vector<PartID>&& pids) {
    RAPIDSMPF_EXPECTS(pids.size() > 0, "insert_finished with empty pids");

    std::vector<detail::ChunkID> expected_num_chunks;
    expected_num_chunks.reserve(pids.size());

    // collect expected number of chunks for each partition
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        for (auto pid : pids) {
            expected_num_chunks.push_back(outbound_chunk_counter_[pid]);
        }
    }

    // insert a finished chunk for each partition
    for (std::size_t i = 0; i < pids.size(); ++i) {
        insert(
            detail::Chunk::from_finished_partition(
                get_new_cid(), pids[i], expected_num_chunks[i] + 1
            )
        );
    }
}

std::vector<PackedData> Shuffler::extract(PartID pid) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::unique_lock<std::mutex> lock(ready_postbox_spilling_mutex_);

    // Quick return if the partition is empty.
    if (ready_postbox_.is_empty(pid)) {
        return std::vector<PackedData>{};
    }

    auto chunks = ready_postbox_.extract(pid);
    lock.unlock();

    std::vector<PackedData> ret;
    ret.reserve(chunks.size());

    std::ranges::transform(chunks, std::back_inserter(ret), [](auto&& p) -> PackedData {
        return {p.second.release_metadata_buffer(), p.second.release_data_buffer()};
    });

    return ret;
}

bool Shuffler::finished() const {
    return finish_counter_.all_finished() && ready_postbox_.empty();
}

PartID Shuffler::wait_any(std::optional<std::chrono::milliseconds> timeout) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    return finish_counter_.wait_any(std::move(timeout));
}

void Shuffler::wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    finish_counter_.wait_on(pid, std::move(timeout));
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
        std::lock_guard<std::mutex> lock(ready_postbox_spilling_mutex_);
        spilled = postbox_spilling(br_, ready_postbox_, spill_need);
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
    ss << "Shuffler(outgoing=" << outgoing_postbox_ << ", received=" << ready_postbox_
       << ", " << finish_counter_;
    return ss.str();
}

}  // namespace rapidsmpf::shuffler
