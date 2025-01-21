/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>

#include <mpi.h>

#include <cudf/concatenate.hpp>
#include <cudf/detail/contiguous_split.hpp>  // `cudf::detail::pack` (stream ordered version)

#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/communicator/communicator.hpp>
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
    OpID op_id,
    PartID total_num_partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    PartitionOwner partition_owner
)
    : total_num_partitions{total_num_partitions},
      partition_owner{partition_owner},
      stream_{stream},
      br_{br},
      comm_{std::move(comm)},
      op_id_{op_id},
      finish_counter_{
          comm_->nranks(), local_partitions(comm_, total_num_partitions, partition_owner)
      } {
    event_loop_thread_ = std::thread(Shuffler::event_loop, this);
    RAPIDSMP_EXPECTS(br_ != nullptr, "the BufferResource cannot be NULL");
}

Shuffler::~Shuffler() {
    if (active_) {
        shutdown();
    }
}

void Shuffler::shutdown() {
    RAPIDSMP_EXPECTS(active_, "shuffler is inactive");
    auto& log = comm_->logger();
    log.info("Shuffler.shutdown() - initiate");
    event_loop_thread_run_.store(false);
    event_loop_thread_.join();
    log.info("Shuffler.shutdown() - done");
    active_ = false;
}

void Shuffler::insert_into_outbox(detail::Chunk&& chunk) {
    auto& log = comm_->logger();
    log.info("insert_into_outbox: ", chunk);
    auto pid = chunk.pid;
    if (chunk.expected_num_chunks) {
        finish_counter_.move_goalpost(
            comm_->rank(), chunk.pid, chunk.expected_num_chunks
        );
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
        insert_into_outbox(std::move(chunk));
    } else {
        inbox_.insert(std::move(chunk));
    }
}

void Shuffler::insert(PartID pid, cudf::packed_columns&& chunk) {
    insert(detail::Chunk{
        pid,
        get_new_cid(),
        0,
        chunk.gpu_data ? chunk.gpu_data->size() : 0,
        std::move(chunk.metadata),
        br_->move(std::move(chunk.gpu_data), stream_)
    });
}

void Shuffler::insert(std::unordered_map<PartID, cudf::packed_columns>&& chunks) {
    for (auto& [pid, packed_columns] : chunks) {
        insert(pid, std::move(packed_columns));
    }
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
    auto chunks = outbox_.extract(pid);
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

    // TODO: check overbooking, do we need to spill to host memory?
    // if(overbooking > 0) {
    //     spill chunks in inbox_ and outbox_
    // }

    // Move the gpu_data to device memory (copy if necessary).
    for (auto& [_, chunk] : chunks) {
        ret.emplace_back(
            std::move(chunk.metadata),
            br_->move_to_device_buffer(std::move(chunk.gpu_data), stream_, reservation)
        );
    }
    return ret;
}

detail::ChunkID Shuffler::get_new_cid() {
    // Place the counter in the first 38 bits (supports 256G chunks).
    std::uint64_t upper = ++chunk_id_counter_ << 26;
    // and place the rank in last 26 bits (supports 64M ranks).
    std::uint64_t lower = comm_->rank();
    return upper | lower;
}

/**
 * @brief Executes a single iteration of the shuffler's event loop.
 *
 * This function manages the movement of data chunks between ranks in the distributed
 * system, handling tasks such as sending and receiving metadata, GPU data, and readiness
 * messages. It also manages the processing of chunks in transit, both outgoing and
 * incoming, and updates the necessary data structures for further processing.
 *
 * @param self Reference to the `Shuffler` instance that owns the event loop.
 * @param fire_and_forget A vector of ongoing "fire-and-forget" operations (non-blocking
 * sends).
 * @param incoming_chunks A multimap of chunks ready to be received, keyed by the source
 * rank.
 * @param outgoing_chunks A map of chunks ready to be sent, keyed by their unique chunk
 * ID.
 * @param in_transit_chunks A map of chunks currently in transit, keyed by their unique
 * chunk ID.
 * @param in_transit_futures A map of futures corresponding to in-transit chunks, keyed by
 * chunk ID.
 */
void Shuffler::run_event_loop_iteration(
    Shuffler& self,
    std::vector<std::unique_ptr<Communicator::Future>>& fire_and_forget,
    std::multimap<Rank, Chunk>& incoming_chunks,
    std::unordered_map<ChunkID, Chunk>& outgoing_chunks,
    std::unordered_map<ChunkID, Chunk>& in_transit_chunks,
    std::unordered_map<ChunkID, std::unique_ptr<Communicator::Future>>& in_transit_futures
) {
    enum TAG : rapidsmp::detail::TagPrefixT {
        metadata = 1,
        gpu_data = 2,
        ready_for_data = 3
    };

    auto& log = self.comm_->logger();

    // Check for new chunks in the inbox and send off their metadata.
    for (auto&& chunk : self.inbox_.extract_all()) {
        auto dst = self.partition_owner(self.comm_, chunk.pid);
        log.info("send metadata to ", dst, ": ", chunk);
        RAPIDSMP_EXPECTS(dst != self.comm_->rank(), "sending chunk to ourselves");

        fire_and_forget.push_back(self.comm_->send(
            chunk.to_metadata_message(), dst, TAG::metadata, self.stream_, self.br_
        ));
        if (chunk.gpu_data_size > 0) {
            RAPIDSMP_EXPECTS(
                outgoing_chunks.insert({chunk.cid, std::move(chunk)}).second,
                "outgoing chunk already exist"
            );
        }
    }

    // Receive any incoming metadata of remote chunks and place them in
    // `incoming_chunks`.
    while (true) {
        auto const [msg, src] = self.comm_->recv_any(TAG::metadata);
        if (msg) {
            auto chunk = Chunk::from_metadata_message(msg);
            log.info("recv_any from ", src, ": ", chunk);
            RAPIDSMP_EXPECTS(
                self.partition_owner(self.comm_, chunk.pid) == self.comm_->rank(),
                "receiving chunk not owned by us"
            );
            incoming_chunks.insert({src, std::move(chunk)});
        } else {
            break;
        }
    }

    // Pick an incoming chunk's gpu_data to receive.
    //
    // TODO: decide to receive into host or device memory.
    // TODO: pick the incoming chunk based on a strategy. For now, we just pick the
    // first chunk.
    // TODO: handle multiple chunks before continuing.
    if (auto first_chunk = incoming_chunks.begin(); first_chunk != incoming_chunks.end())
    {
        auto [src, chunk] = extract_item(incoming_chunks, first_chunk);
        log.info("picked incoming chunk data from ", src, ": ", chunk);
        // If the chunk contains gpu data, we need to receive it. Otherwise, it goes
        // direct to the outbox.
        if (chunk.gpu_data_size > 0) {
            // Tell the source of the chunk that we are ready to receive it.
            fire_and_forget.push_back(self.comm_->send(
                ReadyForDataMessage{chunk.pid, chunk.cid}.pack(),
                src,
                TAG::ready_for_data,
                self.stream_,
                self.br_
            ));
            // Create a new buffer based on memory type availability and prioritizing
            // device over host memory.
            auto recv_buffer =
                allocate_buffer(chunk.gpu_data_size, self.stream_, self.br_);
            // Setup to receive the chunk into `in_transit_*`.
            auto future = self.comm_->recv(
                src, TAG::gpu_data, std::move(recv_buffer), self.stream_
            );
            RAPIDSMP_EXPECTS(
                in_transit_futures.insert({chunk.cid, std::move(future)}).second,
                "in transit future already exist"
            );
            RAPIDSMP_EXPECTS(
                in_transit_chunks.insert({chunk.cid, std::move(chunk)}).second,
                "in transit chunk already exist"
            );
        } else {
            if (chunk.gpu_data == nullptr) {
                chunk.gpu_data = allocate_buffer(0, self.stream_, self.br_);
            }
            self.insert_into_outbox(std::move(chunk));
        }
    }

    // Receive any incoming ready-for-data messages and start sending the
    // requested data.
    while (true) {
        auto const [msg, src] = self.comm_->recv_any(TAG::ready_for_data);
        if (msg) {
            auto ready_for_data_msg = ReadyForDataMessage::unpack(msg);
            auto chunk = extract_value(outgoing_chunks, ready_for_data_msg.cid);
            log.info(
                "recv_any from ", src, ": ", ready_for_data_msg, ", sending: ", chunk
            );
            if (chunk.gpu_data->mem_type == MemoryType::DEVICE) {
                fire_and_forget.push_back(self.comm_->send(
                    std::move(chunk.gpu_data), src, TAG::gpu_data, self.stream_
                ));
            } else {
                RAPIDSMP_FAIL("Not implemented");
            }
        } else {
            break;
        }
    }

    // Check if any data in transit is finished.
    if (!in_transit_futures.empty()) {
        std::vector<ChunkID> finished = self.comm_->test_some(in_transit_futures);
        for (auto cid : finished) {
            auto chunk = extract_value(in_transit_chunks, cid);
            auto future = extract_value(in_transit_futures, cid);
            chunk.gpu_data = self.comm_->get_gpu_data(std::move(future));
            self.insert_into_outbox(std::move(chunk));
        }
    }

    // Check if we can free some of the outstanding futures.
    if (!fire_and_forget.empty()) {
        std::vector<std::size_t> finished = self.comm_->test_some(fire_and_forget);
        // Sort the indexes into `fire_and_forget` in descending order.
        std::sort(finished.begin(), finished.end(), std::greater<>());
        // And erase from the right.
        for (auto i : finished) {
            fire_and_forget.erase(fire_and_forget.begin() + i);
        }
    }
}

/**
 * @brief Manages the shuffler's main event loop for data processing and communication.
 *
 * The event loop continuously processes data transfer tasks, including sending,
 * receiving, and managing chunks across ranks in the distributed system. The loop runs
 * until all communication tasks are completed and the termination condition is met.
 *
 * @param self Pointer to the `Shuffler` instance running the event loop.
 */
void Shuffler::event_loop(Shuffler* self) {
    auto& log = self->comm_->logger();
    std::vector<std::unique_ptr<Communicator::Future>> fire_and_forget;
    std::multimap<Rank, Chunk> incoming_chunks;
    std::unordered_map<ChunkID, Chunk> outgoing_chunks;
    std::unordered_map<ChunkID, Chunk> in_transit_chunks;
    std::unordered_map<ChunkID, std::unique_ptr<Communicator::Future>> in_transit_futures;

    log.info("event loop - start: ", *self);

    // This thread needs to have a cuda context associated with it.
    // For now, do so by calling cudaFree to initialise the driver.
    RMM_CUDA_TRY(cudaFree(nullptr));
    // Continue the loop until both the "run" flag is false and all
    // ongoing communication is done.
    while (self->event_loop_thread_run_
           || !(
               fire_and_forget.empty() && incoming_chunks.empty()
               && outgoing_chunks.empty() && in_transit_chunks.empty()
               && in_transit_futures.empty() && self->inbox_.empty()
           ))
    {
        run_event_loop_iteration(
            *self,
            fire_and_forget,
            incoming_chunks,
            outgoing_chunks,
            in_transit_chunks,
            in_transit_futures
        );
        std::this_thread::yield();

        // Let's add a short sleep to avoid other threads starving under Valgrind.
        if (is_running_under_valgrind()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    log.info("event loop - shutdown: ", self->str());
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
