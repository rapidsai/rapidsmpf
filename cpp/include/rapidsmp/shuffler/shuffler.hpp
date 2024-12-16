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
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/nvtx.hpp>
#include <rapidsmp/shuffler/chunk.hpp>
#include <rapidsmp/shuffler/postbox.hpp>
#include <rapidsmp/utils.hpp>

/**
 * @namespace rapidsmp::shuffler
 * @brief Shuffler interfaces.
 *
 * A shuffle service for cuDF tables. Use @ref Shuffler to perform a single shuffle.
 */
namespace rapidsmp::shuffler {

/**
 * @namespace rapidsmp::shuffler::detail
 * @brief Shuffler private interfaces.
 *
 * This namespace contains private interfaces for internal workings of the shuffler.
 * These interfaces may change without notice and should not be relied upon directly.
 */
namespace detail {

/**
 * @brief Helper to tally the finish status of a shuffle.
 *
 * The `FinishCounter` class tracks the completion of shuffle operations across different
 * ranks and partitions. Each rank maintains a counter for tracking how many chunks have
 * been received for each partition.
 */
class FinishCounter {
  public:
    /**
     * @brief Construct a finish counter.
     *
     * @param nranks The total number of ranks participating in the shuffle.
     * @param local_partitions The partition IDs local to the current rank.
     */
    FinishCounter(Rank nranks, std::vector<PartID> const& local_partitions)
        : nranks_{nranks} {
        // Initially, none of the partitions are ready to wait on.
        for (auto pid : local_partitions) {
            partitions_ready_to_wait_on_.insert({pid, false});
        }
    }

    /**
     * @brief Move the goalpost for a specific rank and partition.
     *
     * This function sets the number of chunks that need to be received from a specific
     * rank and partition. It should only be called once per rank and partition.
     *
     * @param rank The rank the goalpost is assigned to.
     * @param pid The partition ID the goalpost is assigned to.
     * @param nchunks The number of chunks required.
     *
     * @throw cudf::logic_error If the goalpost is moved more than once for the same rank
     * and partition.
     */
    void move_goalpost(Rank rank, PartID pid, ChunkID nchunks) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& [rank_counter, chunk_goal] = goalposts_[pid];
        RAPIDSMP_EXPECTS(
            rank_counter++ < nranks_, "the goalpost was moved more than one per rank"
        );
        chunk_goal += nchunks;
    }

    /**
     * @brief Add a finished chunk to a partition counter.
     *
     * This function increments the finished chunk counter for a specific partition.
     * When the number of finished chunks matches the goalpost, the partition is marked as
     * finished.
     *
     * @param pid The partition ID to update.
     *
     * @throw cudf::logic_error If the partition has already reached the goalpost.
     */
    void add_finished_chunk(PartID pid) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& finished_chunk = ++finished_chunk_counters_[pid];
        auto& [rank_counter, chunk_goal] = goalposts_[pid];

        // The partition is finished if the goalpost has been set by all ranks
        // and the number of finished chunks has reach the goal.
        if (rank_counter == nranks_) {
            if (finished_chunk == chunk_goal) {
                partitions_ready_to_wait_on_.at(pid) = true;
                cv_.notify_all();
            } else {
                RAPIDSMP_EXPECTS(
                    finished_chunk < chunk_goal, "finished chunk exceeds the goal"
                );
            }
        }
    }

    /**
     * @brief Returns whether all partitions are finished (non-blocking).
     *
     * @return True if all partitions are finished, otherwise false.
     */
    [[nodiscard]] bool all_finished() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return partitions_ready_to_wait_on_.empty();
    }

    /**
     * @brief Returns the partition ID of a finished partition that hasn't been waited on
     * (blocking).
     *
     * This function blocks until a partition is finished and ready to be processed.
     *
     * @return The partition ID of a finished partition.
     *
     * @throw std::out_of_range If all partitions have already been waited on.
     */
    PartID wait_any() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            RAPIDSMP_EXPECTS(
                !partitions_ready_to_wait_on_.empty(),
                "no more partitions to wait on",
                std::out_of_range
            );

            // Find the first ready partition (if any).
            auto it = std::find_if(
                partitions_ready_to_wait_on_.begin(),
                partitions_ready_to_wait_on_.end(),
                [](const auto& item) { return item.second; }
            );
            if (it == partitions_ready_to_wait_on_.end()) {
                // No ready partitions, let's wait.
                cv_.wait(lock);
            } else {
                // We extract the partition to avoid returning the same partition twice.
                return extract_key(partitions_ready_to_wait_on_, it);
            }
        }
    }

    /**
     * @brief Returns a vector of partition ids that are finished and haven't been waited
     * on (blocking).
     *
     * This function blocks until at least one partition is finished and ready to be
     * processed.
     *
     * @note It is the caller's responsibility to process all returned partition IDs.
     *
     * @return vector of finished partitions.
     *
     * @throw std::out_of_range If all partitions have been waited on.
     */
    std::vector<PartID> wait_some() {
        std::unique_lock<std::mutex> lock(mutex_);
        RAPIDSMP_EXPECTS(
            !partitions_ready_to_wait_on_.empty(),
            "no more partitions to wait on",
            std::out_of_range
        );
        cv_.wait(lock, [&]() {
            return std::any_of(
                partitions_ready_to_wait_on_.begin(),
                partitions_ready_to_wait_on_.end(),
                [](auto const& item) { return item.second; }
            );
        });
        std::vector<PartID> result{};
        // TODO: hand-writing iteration rather than range-for to avoid
        // needing to rehash the key during extract_key. Needs
        // std::ranges, I think.
        for (auto it = partitions_ready_to_wait_on_.begin();
             it != partitions_ready_to_wait_on_.end();
             *it++)
        {
            if (it->second) {
                result.push_back(extract_key(partitions_ready_to_wait_on_, it));
            }
        }
        return result;
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;


  private:
    Rank const nranks_;
    // The goalpost of each partition. The goal is a rank counter to track how many ranks
    // has reported their goal, and a chunk counter that specifies the goal. It is only
    // when all ranks has reported their goal that the goalpost is final.
    std::unordered_map<PartID, std::pair<Rank, ChunkID>> goalposts_;
    // The finished chunk counter of each partition. The goal of a partition has been
    // reach when its counter equals the goalpost.
    std::unordered_map<PartID, ChunkID> finished_chunk_counters_;
    // A partition has three states:
    //   - If it is false, the partition isn't finished.
    //   - If it is true, the partition is finished and can be waited on.
    //   - If it is absent, the partition is finished and has already been waited on.
    std::unordered_map<PartID, bool> partitions_ready_to_wait_on_;
    mutable std::mutex mutex_;  // TODO: use a shared_mutex lock?
    mutable std::condition_variable cv_;
};
}  // namespace detail

/**
 * @brief Shuffle service for cuDF tables.
 *
 * The `Shuffler` class provides an interface for performing a shuffle operation on cuDF
 * tables, using a partitioning scheme to distribute and collect data chunks across
 * different ranks.
 */
class Shuffler {
  public:
    /**
     * @brief Function that given a @ref Communicator and a @ref
     * PartID, returns the @ref rapidsmp::Rank of the _owning_ node.
     */
    using PartitionOwner = std::function<Rank(std::shared_ptr<Communicator>, PartID)>;

    /**
     * @brief A @ref PartitionOwner that distribute the partition using round robin.
     *
     * @param comm The communicator to use.
     * @param pid The partition ID to query.
     * @return The rank owning the partition.
     */
    static Rank round_robin(std::shared_ptr<Communicator> const& comm, PartID pid) {
        return pid % comm->nranks();
    }

    /**
     * @brief Returns the local partition IDs owned by the current node.
     *
     * @param comm The communicator to use.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param partition_owner Function that determines partition ownership.
     * @return A vector of partition IDs owned by the current node.
     */
    static std::vector<PartID> local_partitions(
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

    /**
     * @brief Construct a new shuffler for a single shuffle.
     *
     * @param comm The communicator to use.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param partition_owner Function to determine partition ownership.
     * @param stream The CUDA stream for memory operations.
     * @param mr The device memory resource.
     */
    Shuffler(
        std::shared_ptr<Communicator> comm,
        PartID total_num_partitions,
        PartitionOwner partition_owner = round_robin,
        rmm::cuda_stream_view stream = cudf::get_default_stream(),
        BufferResource br = BufferResource(cudf::get_current_device_resource_ref())
    )
        : total_num_partitions{total_num_partitions},
          partition_owner{partition_owner},
          stream_{stream},
          br_{br},
          comm_{std::move(comm)},
          finish_counter_{
              comm_->nranks(),
              local_partitions(comm_, total_num_partitions, partition_owner)
          } {
        event_loop_thread_ = std::thread(Shuffler::event_loop, this);
    }

    ~Shuffler() {
        if (active_) {
            shutdown();
        }
    }

    /**
     * @brief Shutdown the shuffle, blocking until all inflight communication is done.
     *
     * @throw cudf::logic_error If the shuffler is already inactive.
     */
    void shutdown() {
        RAPIDSMP_EXPECTS(active_, "shuffler is inactive");
        auto& log = comm_->logger();
        log.info("Shuffler.shutdown() - initiate");
        event_loop_thread_run_.store(false);
        event_loop_thread_.join();
        log.info("Shuffler.shutdown() - done");
        active_ = false;
    }

  private:
    /**
     * @brief Insert a chunk into the outbox (the chunk is ready for the user).
     *
     * @param chunk The chunk to insert.
     */
    void insert_into_outbox(detail::Chunk&& chunk) {
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

  public:
    /**
     * @brief Insert a chunk into the shuffle.
     *
     * @param chunk The chunk to insert.
     */
    void insert(detail::Chunk&& chunk) {
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

    /**
     * @brief Insert a packed (serialized) chunk into the shuffle.
     *
     * @param pid The partition ID the chunk belong to.
     * @param chunk The packed chunk, `cudf::table`, to insert.
     */
    void insert(PartID pid, cudf::packed_columns&& chunk) {
        insert(detail::Chunk{pid, get_new_cid(), std::move(chunk)});
    }

    /**
     * @brief Insert a bunch of packed (serialized) chunks into the shuffle.
     *
     * @param chunks A map of partition IDs and their packed chunks.
     */
    void insert(std::unordered_map<PartID, cudf::packed_columns>&& chunks) {
        for (auto& [pid, packed_columns] : chunks) {
            insert(pid, std::move(packed_columns));
        }
    }

    /**
     * @brief Insert a finish mark for a partition.
     *
     * This tells the shuffler that no more chunks of the specified partition are coming.
     *
     * @param pid The partition ID to mark as finished.
     */
    void insert_finished(PartID pid) {
        detail::ChunkID expected_num_chunks;
        {
            std::lock_guard const lock(outbound_chunk_counter_mutex_);
            expected_num_chunks = outbound_chunk_counter_[pid];
        }
        insert(detail::Chunk{pid, get_new_cid(), expected_num_chunks + 1});
    }

    /**
     * @brief Extract all chunks of a specific partition.
     *
     * @param pid The partition ID.
     * @return A vector of packed columns (chunks) for the partition.
     */
    [[nodiscard]] std::vector<cudf::packed_columns> extract(PartID pid) {
        auto chunks = outbox_.extract(pid);
        std::vector<cudf::packed_columns> ret;
        ret.reserve(chunks.size());
        for (auto& [_, chunk] : chunks) {
            // TODO: make sure that the gpu_data is on device memory (copy if necessary).
            ret.emplace_back(
                std::move(chunk.metadata), std::move(chunk.gpu_data->device())
            );
        }
        return ret;
    }

    /**
     * @brief Check if all partitions are finished.
     *
     * @return True if all partitions are finished, otherwise false.
     */
    [[nodiscard]] bool finished() const {
        return finish_counter_.all_finished();
    }

    /**
     * @brief Wait for any partition to finish.
     *
     * @return The partition ID of the next finished partition.
     */
    PartID wait_any() {
        RAPIDSMP_NVTX_FUNC_RANGE();
        return finish_counter_.wait_any();
    }

    /**
     * @brief Wait for at least one partition to finish.
     *
     * @return The partition IDs of all finished partitions.
     */
    std::vector<PartID> wait_some() {
        RAPIDSMP_NVTX_FUNC_RANGE();
        return finish_counter_.wait_some();
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    static void event_loop(Shuffler* self);
    static void run_event_loop_iteration(
        Shuffler& self,
        std::vector<std::unique_ptr<Communicator::Future>>& fire_and_forget,
        std::multimap<Rank, detail::Chunk>& incoming_chunks,
        std::unordered_map<detail::ChunkID, detail::Chunk>& outgoing_chunks,
        std::unordered_map<detail::ChunkID, detail::Chunk>& in_transit_chunks,
        std::unordered_map<detail::ChunkID, std::unique_ptr<Communicator::Future>>&
            in_transit_futures
    );

    [[nodiscard]] detail::ChunkID get_new_cid() {
        // Place the counter in the first 38 bits (supports 256G chunks).
        std::uint64_t upper = ++chunk_id_counter_ << 26;
        // and place the rank in last 26 bits (supports 64M ranks).
        std::uint64_t lower = comm_->rank();
        return upper | lower;
    }

  public:
    PartID const total_num_partitions;  ///< Total number of partition in the shuffle.
    PartitionOwner const partition_owner;  ///< Function to determine partition ownership

  private:
    rmm::cuda_stream_view stream_;
    BufferResource br_;
    bool active_{true};
    detail::PostBox inbox_;
    detail::PostBox outbox_;

    std::shared_ptr<Communicator> comm_;
    std::thread event_loop_thread_;
    std::atomic<bool> event_loop_thread_run_{true};

    detail::FinishCounter finish_counter_;
    std::unordered_map<PartID, detail::ChunkID> outbound_chunk_counter_;
    mutable std::mutex outbound_chunk_counter_mutex_;

    std::atomic<detail::ChunkID> chunk_id_counter_{0};
};

/**
 * @brief Overloads the stream insertion operator for the FinishCounter class.
 *
 * This function allows a description of a FinishCounter to be written to an output
 * stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, detail::FinishCounter const& obj) {
    os << obj.str();
    return os;
}

/**
 * @brief Overloads the stream insertion operator for the Shuffler class.
 *
 * This function allows a description of a Shuffler to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, Shuffler const& obj) {
    os << obj.str();
    return os;
}


}  // namespace rapidsmp::shuffler
