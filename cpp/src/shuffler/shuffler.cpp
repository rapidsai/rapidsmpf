/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

#include <mpi.h>

#include <cudf/concatenate.hpp>
#include <cudf/detail/contiguous_split.hpp>  // `cudf::detail::pack` (stream ordered version)

#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp::shuffler {

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
    RAPIDSMP_EXPECTS(reservation.size() == 0, "didn't use all of the reservation");
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
    RAPIDSMP_EXPECTS(
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
std::size_t postbox_spilling(
    BufferResource* br,
    Communicator::Logger& log,
    Statistics& statistics,
    rmm::cuda_stream_view stream,
    PostBox& postbox,
    std::size_t amount
) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    auto const t0_elapsed = Clock::now();
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
        chunk.gpu_data = br->move(
            MemoryType::HOST, std::move(chunk.gpu_data), stream, host_reservation
        );
        postbox.insert(std::move(chunk));
        if ((total_spilled += size) >= amount) {
            break;
        }
    }
    statistics.add_duration_stat("spill-time-device-to-host", Clock::now() - t0_elapsed);
    statistics.add_bytes_stat("spill-bytes-device-to-host", total_spilled);
    if (total_spilled < amount) {
        // TODO: use a "max" statistic when it is available, for now we use the average.
        statistics.add_stat(
            "spill-breach-device-limit",
            amount - total_spilled,
            [](std::ostream& os, std::size_t count, double val) {
                os << "avg " << format_nbytes(val / count);
            }
        );
    }
    return total_spilled;
}
}  // namespace

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
      comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      op_id_{op_id},
      finish_counter_{
          static_cast<Rank>(comm_->nranks()),
          local_partitions(comm_, total_num_partitions, partition_owner)
      },
      statistics_{std::move(statistics)} {
    RAPIDSMP_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be NULL");
    RAPIDSMP_EXPECTS(br_ != nullptr, "the buffer resource pointer cannot be NULL");
    RAPIDSMP_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");

    // Register a spill function that spill buffers in this shuffler.
    // Note, the spill function can use `this` because a Shuffler isn't movable.
    br->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );

    // We need to register the progress function with the progress thread, but
    // that cannot be done in the constructor list because the Shuffler isn't
    // fully constructed yet.
    function_id_ = progress_thread_->add_function([this]() { return progress(); });
}

Shuffler::~Shuffler() {
    shutdown();
}

void Shuffler::shutdown() {
    if (active_.load()) {
        auto& log = comm_->logger();
        log.debug("Shuffler.shutdown() - initiate");
        active_.store(false);
        progress_thread_->remove_function(*function_id_);
        log.debug("Shuffler.shutdown() - done");
    }
}

void Shuffler::insert_into_outbox(detail::Chunk&& chunk) {
    auto& log = comm_->logger();
    log.trace("insert_into_outbox: ", chunk);
    auto pid = chunk.pid;
    if (chunk.expected_num_chunks) {
        finish_counter_.move_goalpost(chunk.pid, chunk.expected_num_chunks);
    } else {
        outbox_.insert(std::move(chunk));
    }
    finish_counter_.add_finished_chunk(pid);
}

void Shuffler::insert(detail::Chunk&& chunk) {
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        ++outbound_chunk_counter_[chunk.pid];
    }
    if (partition_owner(comm_, chunk.pid) == comm_->rank()) {
        if (chunk.gpu_data) {
            statistics_->add_bytes_stat("shuffle-payload-send", chunk.gpu_data->size);
            statistics_->add_bytes_stat("shuffle-payload-recv", chunk.gpu_data->size);
        }
        insert_into_outbox(std::move(chunk));
    } else {
        inbox_.insert(std::move(chunk));
    }
}

void Shuffler::insert(std::unordered_map<PartID, cudf::packed_columns>&& chunks) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    auto& log = comm_->logger();

    // Insert each chunk into the inbox.
    for (auto& [pid, packed_columns] : chunks) {
        // Check if we should spill the chunk before inserting into the inbox.
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        if (headroom < 0 && packed_columns.gpu_data) {
            auto [host_reservation, host_overbooking] =
                br_->reserve(MemoryType::HOST, packed_columns.gpu_data->size(), true);
            if (host_overbooking > 0) {
                log.warn(
                    "Cannot spill to host because of host memory overbooking: ",
                    format_nbytes(host_overbooking)
                );
                continue;
            }
            auto chunk = create_chunk(
                pid,
                std::move(packed_columns.metadata),
                std::move(packed_columns.gpu_data)
            );
            // Spill the new chunk before inserting.
            auto const t0_elapsed = Clock::now();
            chunk.gpu_data = br_->move(
                MemoryType::HOST, std::move(chunk.gpu_data), stream_, host_reservation
            );
            statistics_->add_duration_stat(
                "spill-time-host-to-device", Clock::now() - t0_elapsed
            );
            statistics_->add_bytes_stat(
                "spill-bytes-host-to-device", chunk.gpu_data->size
            );
            insert(std::move(chunk));
        } else {
            insert(create_chunk(
                pid,
                std::move(packed_columns.metadata),
                std::move(packed_columns.gpu_data)
            ));
        }
    }

    // Spill if current available device memory is still negative.
    br_->spill_manager().spill_to_make_headroom(0);
}

void Shuffler::insert_finished(PartID pid) {
    detail::ChunkID expected_num_chunks;
    {
        std::lock_guard const lock(outbound_chunk_counter_mutex_);
        expected_num_chunks = outbound_chunk_counter_[pid];
    }
    insert(detail::Chunk{pid, get_new_cid(), expected_num_chunks + 1});
}

std::vector<cudf::packed_columns> Shuffler::extract(PartID pid) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    // Protect the chunk extraction to make sure we don't get a chunk
    // `Shuffler::spill` is in the process of spilling.
    std::unique_lock<std::mutex> lock(outbox_spilling_mutex_);
    auto chunks = outbox_.extract(pid);
    lock.unlock();
    std::vector<cudf::packed_columns> ret;
    ret.reserve(chunks.size());

    // Sum the total size of all chunks not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, chunk] : chunks) {
        if (chunk.gpu_data->mem_type != MemoryType::DEVICE) {
            non_device_size += chunk.gpu_data->size;
        }
    }
    // This total sum is what we need to reserve before moving them to device.
    auto [reservation, overbooking] =
        br_->reserve(MemoryType::DEVICE, non_device_size, true);

    // Check overbooking, do we need to spill to host memory?
    if (overbooking > 0) {
        spill(overbooking);
    }

    // Move the gpu_data to device memory (copy if necessary).
    auto const t0_unspill = Clock::now();
    std::uint64_t total_unspilled{0};
    for (auto& [_, chunk] : chunks) {
        if (chunk.gpu_data->mem_type != MemoryType::DEVICE) {
            total_unspilled += chunk.gpu_data->size;
        }
        ret.emplace_back(
            std::move(chunk.metadata),
            br_->move_to_device_buffer(std::move(chunk.gpu_data), stream_, reservation)
        );
    }
    statistics_->add_duration_stat(
        "spill-time-host-to-device", Clock::now() - t0_unspill
    );
    statistics_->add_bytes_stat("spill-bytes-host-to-device", total_unspilled);
    return ret;
}

std::size_t Shuffler::spill(std::optional<std::size_t> amount) {
    RAPIDSMP_NVTX_FUNC_RANGE();
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
        std::lock_guard<std::mutex> lock(outbox_spilling_mutex_);
        spilled = postbox_spilling(
            br_, comm_->logger(), *statistics_, stream_, outbox_, spill_need
        );
    }
    return spilled;
}

detail::ChunkID Shuffler::get_new_cid() {
    // Place the counter in the first 38 bits (supports 256G chunks).
    std::uint64_t upper = ++chunk_id_counter_ << 26;
    // and place the rank in last 26 bits (supports 64M ranks).
    std::uint64_t lower = static_cast<std::uint64_t>(comm_->rank());
    return upper | lower;
}

ProgressThread::ProgressState Shuffler::progress() {
    auto const t0_event_loop = Clock::now();

    // Tags for each stage of the shuffle
    Tag const ready_for_data_tag{op_id_, 1};
    Tag const metadata_tag{op_id_, 2};
    Tag const gpu_data_tag{op_id_, 3};

    auto& log = comm_->logger();
    auto& stats = *statistics_;

    // Check for new chunks in the inbox and send off their metadata.
    auto const t0_send_metadata = Clock::now();
    for (auto&& chunk : inbox_.extract_all()) {
        auto dst = partition_owner(comm_, chunk.pid);
        log.trace("send metadata to ", dst, ": ", chunk);
        RAPIDSMP_EXPECTS(dst != comm_->rank(), "sending chunk to ourselves");

        fire_and_forget_.push_back(
            comm_->send(chunk.to_metadata_message(), dst, metadata_tag, br_)
        );
        if (chunk.gpu_data_size > 0) {
            RAPIDSMP_EXPECTS(
                outgoing_chunks_.insert({chunk.cid, std::move(chunk)}).second,
                "outgoing chunk already exist"
            );
        }
    }
    stats.add_duration_stat("event-loop-metadata-send", Clock::now() - t0_send_metadata);

    // Receive any incoming metadata of remote chunks and place them in
    // `incoming_chunks_`.
    auto const t0_metadata_recv = Clock::now();
    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag);
        if (msg) {
            auto chunk = Chunk::from_metadata_message(msg);
            log.trace("recv_any from ", src, ": ", chunk);
            RAPIDSMP_EXPECTS(
                partition_owner(comm_, chunk.pid) == comm_->rank(),
                "receiving chunk not owned by us"
            );
            incoming_chunks_.insert({src, std::move(chunk)});
        } else {
            break;
        }
    }
    stats.add_duration_stat("event-loop-metadata-recv", Clock::now() - t0_metadata_recv);

    // Post receives for incoming chunks
    auto const t0_post_incoming_chunk_recv = Clock::now();
    for (auto first_chunk = incoming_chunks_.begin();
         first_chunk != incoming_chunks_.end();)
    {
        // Note, extract_item invalidates the iterator, so must increment here.
        auto [src, chunk] = extract_item(incoming_chunks_, first_chunk++);
        log.trace("picked incoming chunk data from ", src, ": ", chunk);
        // If the chunk contains gpu data, we need to receive it. Otherwise, it goes
        // directly to the outbox.
        if (chunk.gpu_data_size > 0) {
            // Create a new buffer and let the buffer resource decide the memory type.
            auto recv_buffer = allocate_buffer(chunk.gpu_data_size, stream_, br_);
            if (recv_buffer->mem_type == MemoryType::HOST) {
                stats.add_bytes_stat("spill-bytes-recv-to-host", recv_buffer->size);
            }

            // Setup to receive the chunk into `in_transit_*`.
            auto future = comm_->recv(src, gpu_data_tag, std::move(recv_buffer));
            RAPIDSMP_EXPECTS(
                in_transit_futures_.insert({chunk.cid, std::move(future)}).second,
                "in transit future already exist"
            );
            RAPIDSMP_EXPECTS(
                in_transit_chunks_.insert({chunk.cid, std::move(chunk)}).second,
                "in transit chunk already exist"
            );
            statistics_->add_bytes_stat("shuffle-payload-recv", chunk.gpu_data_size);
            // Tell the source of the chunk that we are ready to receive it.
            fire_and_forget_.push_back(comm_->send(
                ReadyForDataMessage{chunk.pid, chunk.cid}.pack(),
                src,
                ready_for_data_tag,
                br_
            ));
        } else {
            if (chunk.gpu_data == nullptr) {
                chunk.gpu_data = allocate_buffer(0, stream_, br_);
            }
            insert_into_outbox(std::move(chunk));
        }
    }
    stats.add_duration_stat(
        "event-loop-post-incoming-chunk-recv", Clock::now() - t0_post_incoming_chunk_recv
    );

    // Receive any incoming ready-for-data messages and start sending the
    // requested data.
    auto const t0_init_gpu_data_send = Clock::now();
    while (true) {
        auto const [msg, src] = comm_->recv_any(ready_for_data_tag);
        if (msg) {
            auto ready_for_data_msg = ReadyForDataMessage::unpack(msg);
            auto chunk = extract_value(outgoing_chunks_, ready_for_data_msg.cid);
            log.trace(
                "recv_any from ", src, ": ", ready_for_data_msg, ", sending: ", chunk
            );
            statistics_->add_bytes_stat("shuffle-payload-send", chunk.gpu_data->size);
            fire_and_forget_.push_back(
                comm_->send(std::move(chunk.gpu_data), src, gpu_data_tag)
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
        std::vector<ChunkID> finished = comm_->test_some(in_transit_futures_);
        for (auto cid : finished) {
            auto chunk = extract_value(in_transit_chunks_, cid);
            auto future = extract_value(in_transit_futures_, cid);
            chunk.gpu_data = comm_->get_gpu_data(std::move(future));
            insert_into_outbox(std::move(chunk));
        }
    }

    // Check if we can free some of the outstanding futures.
    if (!fire_and_forget_.empty()) {
        std::vector<std::size_t> finished = comm_->test_some(fire_and_forget_);
        if (!finished.empty()) {
            // Sort the indexes into `fire_and_forget` in descending order.
            std::sort(finished.begin(), finished.end(), std::greater<>());
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

    // Only return Done if the shuffler is still active (shutdown() was not called)
    // and all containers are empty.
    return (active_.load()
            || !(
                fire_and_forget_.empty() && incoming_chunks_.empty()
                && outgoing_chunks_.empty() && in_transit_chunks_.empty()
                && in_transit_futures_.empty() && inbox_.empty()
            ))
               ? ProgressThread::ProgressState::InProgress
               : ProgressThread::ProgressState::Done;
}

std::string Shuffler::str() const {
    std::stringstream ss;
    ss << "Shuffler(inbox=" << inbox_ << ", outbox=" << outbox_ << ", "
       << finish_counter_;
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

}  // namespace rapidsmp::shuffler
