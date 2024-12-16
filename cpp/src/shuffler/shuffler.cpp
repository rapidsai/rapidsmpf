/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp::shuffler {

using namespace detail;

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
    enum TAG : int {
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
            chunk.to_metadata_message(), dst, TAG::metadata, self.stream_, self.mr_
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
                self.mr_
            ));
            // Setup to receive the chunk into `in_transit_*`.
            auto future = self.comm_->recv(
                src, TAG::gpu_data, chunk.gpu_data_size, self.stream_, self.mr_
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
                chunk.gpu_data = std::make_unique<Buffer>(
                    std::make_unique<rmm::device_buffer>(0, self.stream_, self.mr_)
                );
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
            if (chunk.gpu_data->mem_type == MemoryType::device) {
                fire_and_forget.push_back(self.comm_->send(
                    std::move(chunk.gpu_data->device()),
                    src,
                    TAG::gpu_data,
                    self.stream_,
                    self.mr_
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
            chunk.gpu_data =
                self.comm_->get_gpu_data(std::move(future), self.stream_, self.mr_);
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
