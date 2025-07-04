/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler {

using namespace detail;

namespace {

/**
 * @brief Help function to reserve and allocate a new buffer.
 *
 * First reserve the memory type and then use the reservation to allocate a new
 * buffer. Returns null if reservation failed.
 *
 * @param mem_type The target memory type.
 * @param size The size of the buffer in bytes.
 * @param stream CUDA stream to use for device allocations.
 * @param br Buffer resource used for the reservation and allocation.
 * @returns A new buffer or nullptr.
 */
std::unique_ptr<Buffer> allocate_buffer(
    MemoryType mem_type,
    std::size_t size,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto [reservation, _] = br->reserve(mem_type, size, false);
    if (reservation.size() != size) {
        return nullptr;
    }
    auto ret = br->allocate(mem_type, size, stream, reservation);
    RAPIDSMPF_EXPECTS(reservation.size() == 0, "didn't use all of the reservation");
    return ret;
}

/**
 * @brief Help function to reserve and allocate a new buffer.
 *
 * First reserve device memory and then use the reservation to allocate a new
 * buffer. If not enough device memory is available, host memory is reserved and
 * allocated instead.
 *
 * @param size The size of the buffer in bytes.
 * @param stream CUDA stream to use for device allocations.
 * @param br Buffer resource used for the reservation and allocation.
 * @returns A new buffer.
 *
 * @throws std::overflow_error if both the reservation of device and host memory
 * failed.
 */
std::unique_ptr<Buffer> allocate_buffer(
    std::size_t size, rmm::cuda_stream_view stream, BufferResource* br
) {
    std::unique_ptr<Buffer> ret = allocate_buffer(MemoryType::DEVICE, size, stream, br);
    if (ret) {
        return ret;
    }
    // If not enough device memory is available, we try host memory.
    ret = allocate_buffer(MemoryType::HOST, size, stream, br);
    RAPIDSMPF_EXPECTS(
        ret,
        "Cannot reserve " + format_nbytes(size) + " of device or host memory",
        std::overflow_error
    );
    return ret;
}

/**
 * @brief Spills memory buffers within a postbox, e.g., from device to host memory.
 *
 * This function moves a specified amount of memory from device to host storage
 * or another lower-priority memory space, helping manage limited GPU memory
 * by offloading excess data.
 *
 * @note While spilling, chunks are temporarily extracted from the postbox thus other
 * threads trying to extract a chunk that is in the process of being spilled, will fail.
 * To avoid this, the Shuffler uses `outbox_spillling_mutex_` to serialize extractions.
 *
 * @param br Buffer resource for memory allocation.
 * @param log A logger for recording events and debugging information.
 * @param statistics The statistics instance to use.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param postbox The PostBox containing buffers to be spilled.
 * @param amount The maximum amount of data (in bytes) to be spilled.
 *
 * @return The actual amount of data successfully spilled from the postbox.
 */
template <typename KeyType>
std::size_t postbox_spilling(
    BufferResource* br,
    Communicator::Logger& log,
    rmm::cuda_stream_view stream,
    PostBox<KeyType>& postbox,
    std::size_t amount
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    // Let's look for chunks to spill in the outbox.
    auto const chunk_info = postbox.search(MemoryType::DEVICE);
    std::size_t total_spilled{0};
    for (auto [pid, cid, size] : chunk_info) {
        // TODO: Use a clever strategy to decide which chunks to spill. For now, we
        // just spill the chunks in an arbitrary order.
        auto [host_reservation, host_overbooking] =
            br->reserve(MemoryType::HOST, size, true);
        if (host_overbooking > 0) {
            log.warn(
                "Cannot spill to host because of host memory overbooking: ",
                format_nbytes(host_overbooking)
            );
            continue;
        }
        // We extract the chunk, spilled it, and insert it back into the PostBox.
        auto chunk = postbox.extract(pid, cid);
        chunk.set_data_buffer(br->move(
            MemoryType::HOST, chunk.release_data_buffer(), stream, host_reservation
        ));
        postbox.insert(std::move(chunk));
        if ((total_spilled += size) >= amount) {
            break;
        }
    }
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
     * system, handling tasks such as sending and receiving metadata, GPU data, and
     * readiness messages. It also manages the processing of chunks in transit, both
     * outgoing and incoming, and updates the necessary data structures for further
     * processing.
     *
     * @return The progress state of the shuffler.
     */
    ProgressThread::ProgressState operator()() {
        auto const t0_event_loop = Clock::now();

        // Tags for each stage of the shuffle
        Tag const ready_for_data_tag{shuffler_.op_id_, 1};
        Tag const metadata_tag{shuffler_.op_id_, 2};
        Tag const gpu_data_tag{shuffler_.op_id_, 3};

        auto& log = shuffler_.comm_->logger();
        auto& stats = *shuffler_.statistics_;

        // Check for new chunks in the inbox and send off their metadata.
        auto const t0_send_metadata = Clock::now();
        for (auto&& chunk : shuffler_.outgoing_postbox_.extract_all_ready()) {
            // All messages in the chunk maps to the same key (checked by the PostBox)
            // thus we can use the partition ID of the first message in the chunk to
            // determine the source rank of all of them.
            auto dst = shuffler_.partition_owner(shuffler_.comm_, chunk.part_id(0));
            log.trace("send metadata to ", dst, ": ", chunk);
            RAPIDSMPF_EXPECTS(
                dst != shuffler_.comm_->rank(), "sending chunk to ourselves"
            );

            fire_and_forget_.push_back(
                shuffler_.comm_->send(chunk.serialize(), dst, metadata_tag, shuffler_.br_)
            );
            if (chunk.concat_data_size() > 0) {
                RAPIDSMPF_EXPECTS(
                    outgoing_chunks_.insert({chunk.chunk_id(), std::move(chunk)}).second,
                    "outgoing chunk already exist"
                );
            }
        }
        stats.add_duration_stat(
            "event-loop-metadata-send", Clock::now() - t0_send_metadata
        );

        // Receive any incoming metadata of remote chunks and place them in
        // `incoming_chunks_`.
        auto const t0_metadata_recv = Clock::now();
        while (true) {
            auto const [msg, src] = shuffler_.comm_->recv_any(metadata_tag);
            if (msg) {
                auto chunk = Chunk::deserialize(*msg, false);
                log.trace("recv_any from ", src, ": ", chunk);
                // All messages in the chunk maps to the same Rank (checked by the
                // PostBox) thus we can use the partition ID of the first message in the
                // chunk to determine the source rank of all of them.
                RAPIDSMPF_EXPECTS(
                    shuffler_.partition_owner(shuffler_.comm_, chunk.part_id(0))
                        == shuffler_.comm_->rank(),
                    "receiving chunk not owned by us"
                );
                incoming_chunks_.insert({src, std::move(chunk)});
            } else {
                break;
            }
        }
        stats.add_duration_stat(
            "event-loop-metadata-recv", Clock::now() - t0_metadata_recv
        );

        // Post receives for incoming chunks
        auto const t0_post_incoming_chunk_recv = Clock::now();
        for (auto it = incoming_chunks_.begin(); it != incoming_chunks_.end();) {
            auto& [src, chunk] = *it;
            log.trace("checking incoming chunk data from ", src, ": ", chunk);

            // If the chunk contains gpu data, we need to receive it. Otherwise, it goes
            // directly to the ready postbox.
            if (chunk.concat_data_size() > 0) {
                if (!chunk.is_data_buffer_set()) {
                    // Create a new buffer and let the buffer resource decide the memory
                    // type.
                    chunk.set_data_buffer(allocate_buffer(
                        chunk.concat_data_size(), shuffler_.stream_, shuffler_.br_
                    ));
                    if (chunk.data_memory_type() == MemoryType::HOST) {
                        stats.add_bytes_stat(
                            "spill-bytes-recv-to-host", chunk.concat_data_size()
                        );
                    }
                }

                // Check if the buffer is ready to be used
                if (!chunk.is_ready()) {
                    // Buffer is not ready yet, skip to next item
                    ++it;
                    continue;
                }

                // At this point we know we can process this item, so extract it.
                // Note: extract_item invalidates the iterator, so must increment here.
                auto [src, chunk] = extract_item(incoming_chunks_, it++);

                // Setup to receive the chunk into `in_transit_*`.
                // transfer the data buffer from the chunk to the future
                auto future =
                    shuffler_.comm_->recv(src, gpu_data_tag, chunk.release_data_buffer());
                RAPIDSMPF_EXPECTS(
                    in_transit_futures_.insert({chunk.chunk_id(), std::move(future)})
                        .second,
                    "in transit future already exist"
                );
                RAPIDSMPF_EXPECTS(
                    in_transit_chunks_.insert({chunk.chunk_id(), std::move(chunk)})
                        .second,
                    "in transit chunk already exist"
                );
                shuffler_.statistics_->add_bytes_stat(
                    "shuffle-payload-recv", chunk.concat_data_size()
                );
                // Tell the source of the chunk that we are ready to receive it.
                // All partition IDs in the chunk must map to the same key (rank).
                fire_and_forget_.push_back(shuffler_.comm_->send(
                    ReadyForDataMessage{chunk.chunk_id()}.pack(),
                    src,
                    ready_for_data_tag,
                    shuffler_.br_
                ));
            } else {  // chunk contains control messages and/or metadata-only messages
                // At this point we know we can process this item, so extract it.
                // Note: extract_item invalidates the iterator, so must increment here.
                auto [src, chunk] = extract_item(incoming_chunks_, it++);

                // iterate over all messages in the chunk
                for (size_t i = 0; i < chunk.n_messages(); ++i) {
                    auto chunk_copy = chunk.get_data(
                        shuffler_.get_new_cid(), i, shuffler_.stream_, shuffler_.br_
                    );
                    shuffler_.insert_into_ready_postbox(std::move(chunk_copy));
                }
            }
        }

        stats.add_duration_stat(
            "event-loop-post-incoming-chunk-recv",
            Clock::now() - t0_post_incoming_chunk_recv
        );

        // Receive any incoming ready-for-data messages and start sending the
        // requested data.
        auto const t0_init_gpu_data_send = Clock::now();
        while (true) {
            auto const [msg, src] = shuffler_.comm_->recv_any(ready_for_data_tag);
            if (msg) {
                auto ready_for_data_msg = ReadyForDataMessage::unpack(msg);
                auto chunk = extract_value(outgoing_chunks_, ready_for_data_msg.cid);
                log.trace(
                    "recv_any from ", src, ": ", ready_for_data_msg, ", sending: ", chunk
                );
                shuffler_.statistics_->add_bytes_stat(
                    "shuffle-payload-send", chunk.concat_data_size()
                );
                fire_and_forget_.push_back(
                    shuffler_.comm_->send(chunk.release_data_buffer(), src, gpu_data_tag)
                );
            } else {
                break;
            }
        }
        stats.add_duration_stat(
            "event-loop-init-gpu-data-send", Clock::now() - t0_init_gpu_data_send
        );

        // Check if any data in transit is finished.
        auto const t0_check_future_finish = Clock::now();
        if (!in_transit_futures_.empty()) {
            std::vector<ChunkID> finished =
                shuffler_.comm_->test_some(in_transit_futures_);
            for (auto cid : finished) {
                auto chunk = extract_value(in_transit_chunks_, cid);
                auto future = extract_value(in_transit_futures_, cid);
                chunk.set_data_buffer(shuffler_.comm_->get_gpu_data(std::move(future)));

                for (size_t i = 0; i < chunk.n_messages(); ++i) {
                    shuffler_.insert_into_ready_postbox(chunk.get_data(
                        shuffler_.get_new_cid(), i, shuffler_.stream_, shuffler_.br_
                    ));
                }
            }
        }

        // Check if we can free some of the outstanding futures.
        if (!fire_and_forget_.empty()) {
            std::vector<std::size_t> finished =
                shuffler_.comm_->test_some(fire_and_forget_);
            if (!finished.empty()) {
                // Sort the indexes into `fire_and_forget` in descending order.
                std::ranges::sort(finished, std::greater<>());
                // And erase from the right.
                for (auto i : finished) {
                    fire_and_forget_.erase(
                        fire_and_forget_.begin() + static_cast<std::ptrdiff_t>(i)
                    );
                }
            }
        }
        stats.add_duration_stat(
            "event-loop-check-future-finish", Clock::now() - t0_check_future_finish
        );

        stats.add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);

        // Return Done only if the shuffler is inactive (shutdown was called) _and_
        // all containers are empty (all work is done).
        return (shuffler_.active_
                || !(
                    fire_and_forget_.empty() && incoming_chunks_.empty()
                    && outgoing_chunks_.empty() && in_transit_chunks_.empty()
                    && in_transit_futures_.empty() && shuffler_.outgoing_postbox_.empty()
                ))
                   ? ProgressThread::ProgressState::InProgress
                   : ProgressThread::ProgressState::Done;
    }

  private:
    Shuffler& shuffler_;
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::multimap<Rank, detail::Chunk>
        incoming_chunks_;  ///< Chunks ready to be received.
    std::unordered_map<detail::ChunkID, detail::Chunk>
        outgoing_chunks_;  ///< Chunks ready to be sent.
    std::unordered_map<detail::ChunkID, detail::Chunk>
        in_transit_chunks_;  ///< Chunks currently in transit.
    std::unordered_map<detail::ChunkID, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit chunks.
};

std::vector<PartID> Shuffler::local_partitions(
    std::shared_ptr<Communicator> const& comm,
    PartID total_num_partitions,
    PartitionOwner partition_owner
) {
    std::vector<PartID> ret;
    for (PartID i = 0; i < total_num_partitions; ++i) {
        if (partition_owner(comm, i) == comm->rank()) {
            ret.push_back(i);
        }
    }
    return ret;
}

Shuffler::Shuffler(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    PartID total_num_partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    PartitionOwner partition_owner
)
    : total_num_partitions{total_num_partitions},
      partition_owner{partition_owner},
      stream_{stream},
      br_{br},
      outgoing_postbox_{
          [this](PartID pid) -> Rank {
              return this->partition_owner(this->comm_, pid);
          },  // extract Rank from pid
          static_cast<std::size_t>(comm->nranks())
      },
      ready_postbox_{
          [](PartID pid) -> PartID { return pid; },  // identity mapping
          static_cast<std::size_t>(total_num_partitions),
      },
      comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      op_id_{op_id},
      finish_counter_{
          static_cast<Rank>(comm_->nranks()),
          local_partitions(comm_, total_num_partitions, partition_owner)
      },
      statistics_{std::move(statistics)} {
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");

    // We need to register the progress function with the progress thread, but
    // that cannot be done in the constructor's initializer list because the
    // Shuffler isn't fully constructed yet.
    // NB: this only works because `Shuffler` is not movable, otherwise if moved,
    // `this` will become invalid.
    progress_thread_function_id_ =
        progress_thread_->add_function([progress = std::make_shared<Progress>(*this)]() {
            return (*progress)();
        });

    // Register a spill function that spill buffers in this shuffler.
    // Note, the spill function can use `this` because a Shuffler isn't movable.
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );
}

Shuffler::~Shuffler() {
    shutdown();
}

void Shuffler::shutdown() {
    if (active_) {
        auto& log = comm_->logger();
        log.debug("Shuffler.shutdown() - initiate");
        active_ = false;
        progress_thread_->remove_function(progress_thread_function_id_);
        br_->spill_manager().remove_spill_function(spill_function_id_);
        log.debug("Shuffler.shutdown() - done");
    }
}

detail::Chunk Shuffler::create_chunk(
    PartID pid, PackedData&& packed_data, std::shared_ptr<Buffer::Event> event
) {
    return detail::Chunk::from_packed_data(
        get_new_cid(), pid, std::move(packed_data), std::move(event), stream_, br_
    );
}

void Shuffler::insert_into_ready_postbox(detail::Chunk&& chunk) {
    auto& log = comm_->logger();
    log.trace("insert_into_outbox: ", chunk);

    // ready postbox only supports single message chunks
    RAPIDSMPF_EXPECTS(
        chunk.n_messages() == 1, "inserting into ready_postbox with multiple messages"
    );

    auto pid = chunk.part_id(0);
    if (chunk.is_control_message(0)) {
        finish_counter_.move_goalpost(pid, chunk.expected_num_chunks(0));
    } else {
        ready_postbox_.insert(std::move(chunk));
    }
    finish_counter_.add_finished_chunk(pid);
}

void Shuffler::insert(detail::Chunk&& chunk) {
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        // There are multiple partitions in the chunk. So, increment the counter for
        // each partition.
        for (size_t i = 0; i < chunk.n_messages(); ++i) {
            ++outbound_chunk_counter_[chunk.part_id(i)];
        }
    }

    Rank p0_target_rank = partition_owner(comm_, chunk.part_id(0));
    if (p0_target_rank == comm_->rank()) {
        // this is a local chunk, so we can insert it into the ready postbox
        if (chunk.is_data_buffer_set()) {
            statistics_->add_bytes_stat("shuffle-payload-send", chunk.concat_data_size());
            statistics_->add_bytes_stat("shuffle-payload-recv", chunk.concat_data_size());
        }
        insert_into_ready_postbox(std::move(chunk));
    } else {
        // this is a remote chunk, so we need to insert it into the outgoing postbox
        // all messages in the chunk must map to the same key (rank)
        for (size_t i = 1; i < chunk.n_messages(); ++i) {
            RAPIDSMPF_EXPECTS(
                partition_owner(comm_, chunk.part_id(i)) == p0_target_rank,
                "chunk contains messages targeting different ranks"
            );
        }

        outgoing_postbox_.insert(std::move(chunk));
    }
}

void Shuffler::insert(std::unordered_map<PartID, PackedData>&& chunks) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto& log = comm_->logger();

    auto event = std::make_shared<Buffer::Event>(stream_);

    // Insert each chunk into the inbox.
    for (auto& [pid, packed_data] : chunks) {
        // Check if we should spill the chunk before inserting into the inbox.
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        if (headroom < 0 && packed_data.gpu_data) {
            auto [host_reservation, host_overbooking] =
                br_->reserve(MemoryType::HOST, packed_data.gpu_data->size(), true);
            if (host_overbooking > 0) {
                log.warn(
                    "Cannot spill to host because of host memory overbooking: ",
                    format_nbytes(host_overbooking)
                );
                continue;
            }
            auto chunk = create_chunk(pid, std::move(packed_data), event);
            // Spill the new chunk before inserting.
            auto const t0_elapsed = Clock::now();
            chunk.set_data_buffer(br_->move(
                MemoryType::HOST, chunk.release_data_buffer(), stream_, host_reservation
            ));
            statistics_->add_duration_stat(
                "spill-time-device-to-host", Clock::now() - t0_elapsed
            );
            statistics_->add_bytes_stat(
                "spill-bytes-device-to-host", chunk.concat_data_size()
            );
            insert(std::move(chunk));
        } else {
            insert(create_chunk(pid, std::move(packed_data), event));
        }
    }

    // Spill if current available device memory is still negative.
    br_->spill_manager().spill_to_make_headroom(0);
}

void Shuffler::concat_insert(std::unordered_map<PartID, PackedData>&& chunks) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto& log = comm_->logger();

    // this is the amount of memory available on the device.
    int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();

    if (headroom <= 0) {
        log.warn("No memory available for concatenating data. Falling back to insert.");
        return insert(std::move(chunks));
    }

    // TODO handle spilling

    // Create a chunk group for each rank.
    std::vector<std::vector<Chunk>> chunk_groups(size_t(comm_->nranks()));
    // reserve space for each group assuming an even distribution of chunks
    for (auto&& group : chunk_groups) {
        group.reserve(chunks.size() / size_t(comm_->nranks()));
    }

    // total size of data staged in all builders
    int64_t total_staged_data_ = 0;
    auto init_event = std::make_shared<Buffer::Event>(stream_);

    auto build_all_groups_and_insert = [&]() {
        for (auto&& group : chunk_groups) {
            if (!group.empty()) {
                insert(Chunk::concat(std::move(group), get_new_cid(), stream_, br_));
            }
        }
    };

    bool all_groups_built_flag = false;
    for (auto& [pid, packed_data] : chunks) {
        Rank target_rank = partition_owner(comm_, pid);

        // if the chunk is local, do not concatenate
        if (target_rank == comm_->rank()) {
            // no builder for local chunks
            insert(create_chunk(pid, std::move(packed_data), init_event));
            continue;
        }

        // if the packed data size + total_staged_data_ > headroom, no room to add more
        // chunks any of the builders. So, call build on all builders.
        if (!all_groups_built_flag
            && (int64_t(packed_data.gpu_data->size()) + total_staged_data_ > headroom))
        {
            build_all_groups_and_insert();
            all_groups_built_flag = true;
        }

        if (all_groups_built_flag) {
            // insert this chunk without concatenating
            insert(create_chunk(pid, std::move(packed_data), init_event));
        } else {
            // insert this chunk into the builder
            total_staged_data_ += packed_data.gpu_data->ssize();
            chunk_groups[size_t(target_rank)].emplace_back(
                create_chunk(pid, std::move(packed_data), init_event)
            );
        }
    }

    // build any remaining chunks
    if (!all_groups_built_flag) {
        build_all_groups_and_insert();
    }
}

void Shuffler::insert_finished(PartID pid) {
    insert_finished(std::vector<PartID>{pid});
}

void Shuffler::insert_finished(std::vector<PartID>&& pids) {
    RAPIDSMPF_EXPECTS(pids.size() > 0, "insert_finished with empty pids");

    std::vector<detail::ChunkID> expected_num_chunks;
    expected_num_chunks.reserve(pids.size());

    // collect expected number of chunks for each rank
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        for (auto pid : pids) {
            expected_num_chunks.push_back(outbound_chunk_counter_[pid]);
        }
    }

    // if pids only contains one element, we can just insert the finished chunk
    if (pids.size() == 1) {
        insert(detail::Chunk::from_finished_partition(
            get_new_cid(), pids[0], expected_num_chunks[0] + 1
        ));
        return;
    }

    // Create a chunk group for each rank.
    std::vector<std::vector<Chunk>> chunk_groups(size_t(comm_->nranks()));
    // reserve space for each group assuming an even distribution of chunks
    for (auto&& group : chunk_groups) {
        group.reserve(pids.size() / size_t(comm_->nranks()));
    }

    for (size_t i = 0; i < pids.size(); ++i) {
        Rank target_rank = partition_owner(comm_, pids[i]);

        if (target_rank == comm_->rank()) {  // no group for local chunks
            insert(Chunk::from_finished_partition(
                get_new_cid(), pids[i], expected_num_chunks[i] + 1
            ));
        } else {
            chunk_groups[size_t(target_rank)].emplace_back(Chunk::from_finished_partition(
                get_new_cid(), pids[i], expected_num_chunks[i] + 1
            ));
        }
    }

    for (auto&& group : chunk_groups) {
        if (!group.empty()) {
            insert(Chunk::concat(std::move(group), get_new_cid(), stream_, br_));
        }
    }
}

std::vector<PackedData> Shuffler::extract(PartID pid) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    // Protect the chunk extraction to make sure we don't get a chunk
    // `Shuffler::spill` is in the process of spilling.
    std::unique_lock<std::mutex> lock(ready_postbox_spilling_mutex_);
    auto chunks = ready_postbox_.extract(pid);
    lock.unlock();
    std::vector<PackedData> ret;
    ret.reserve(chunks.size());

    // Sum the total size of all chunks not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, chunk] : chunks) {
        if (chunk.data_memory_type() != MemoryType::DEVICE) {
            non_device_size += chunk.concat_data_size();
        }
    }
    // This total sum is what we need to reserve before moving them to device.
    auto [reservation, overbooking] =
        br_->reserve(MemoryType::DEVICE, non_device_size, true);

    // Check overbooking, do we need to spill to host memory?
    if (overbooking > 0) {
        br_->spill_manager().spill(overbooking);
    }

    // Move the data to device memory (copy if necessary).
    auto const t0_unspill = Clock::now();
    std::uint64_t total_unspilled{0};
    for (auto& [_, chunk] : chunks) {
        if (chunk.data_memory_type() != MemoryType::DEVICE) {
            total_unspilled += chunk.concat_data_size();
        }
        ret.emplace_back(
            chunk.release_metadata_buffer(),
            br_->move_to_device_buffer(chunk.release_data_buffer(), stream_, reservation)
        );
    }
    statistics_->add_duration_stat(
        "spill-time-host-to-device", Clock::now() - t0_unspill
    );
    statistics_->add_bytes_stat("spill-bytes-host-to-device", total_unspilled);
    return ret;
}

std::size_t Shuffler::spill(std::optional<std::size_t> amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        if (headroom < 0) {
            spill_need = static_cast<std::size_t>(std::abs(headroom));
        }
    }
    std::size_t spilled{0};
    if (spill_need > 0) {
        std::lock_guard<std::mutex> lock(ready_postbox_spilling_mutex_);
        spilled =
            postbox_spilling(br_, comm_->logger(), stream_, ready_postbox_, spill_need);
    }
    return spilled;
}

detail::ChunkID Shuffler::get_new_cid() {
    // Place the counter in the first 38 bits (supports 256G chunks).
    std::uint64_t upper = ++chunk_id_counter_ << 26;
    // and place the rank in last 26 bits (supports 64M ranks).
    auto lower = static_cast<std::uint64_t>(comm_->rank());
    return upper | lower;
}

std::string Shuffler::str() const {
    std::stringstream ss;
    ss << "Shuffler(outgoing=" << outgoing_postbox_ << ", received=" << ready_postbox_
       << ", " << finish_counter_;
    return ss.str();
}

std::string detail::FinishCounter::str() const {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "FinishCounter(goalposts={";
    for (auto const& [pid, goal] : goalposts_) {
        ss << "p" << pid << ": (" << goal.first << ", " << goal.second << "), ";
    }
    ss << (goalposts_.empty() ? "}" : "\b\b}");
    ss << ", finished={";
    for (auto const& [pid, counter] : finished_chunk_counters_) {
        ss << "p" << pid << ": " << counter << ", ";
    }
    ss << (finished_chunk_counters_.empty() ? "}" : "\b\b}");
    ss << ", partitions_ready_to_wait_on={";
    for (auto const& [pid, finished] : partitions_ready_to_wait_on_) {
        ss << "p" << pid << ": " << finished << ", ";
    }
    ss << (partitions_ready_to_wait_on_.empty() ? "}" : "\b\b}");
    ss << ")";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler
