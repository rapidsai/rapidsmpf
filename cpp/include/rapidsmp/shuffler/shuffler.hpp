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
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
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
#include <rapidsmp/shuffler/finish_counter.hpp>
#include <rapidsmp/shuffler/postbox.hpp>
#include <rapidsmp/utils.hpp>

/**
 * @namespace rapidsmp::shuffler
 * @brief Shuffler interfaces.
 *
 * A shuffle service for cuDF tables. Use `Shuffler` to perform a single shuffle.
 */
namespace rapidsmp::shuffler {

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
     * @brief Function that given a `Communicator` and a `PartID`, returns the
     * `rapidsmp::Rank` of the _owning_ node.
     */
    using PartitionOwner = std::function<Rank(std::shared_ptr<Communicator>, PartID)>;

    /**
     * @brief A `PartitionOwner` that distribute the partition using round robin.
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
    );

    /**
     * @brief Construct a new shuffler for a single shuffle.
     *
     * @param comm The communicator to use.
     * @param op_id The operation ID of the shuffle. This ID is unique for this operation,
     * and should not be reused until all nodes has called `Shuffler::shutdown()`.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param stream The CUDA stream for memory operations.
     * @param br Buffer resource used to allocate temporary and the shuffle result.
     * @param partition_owner Function to determine partition ownership.
     */
    Shuffler(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        PartID total_num_partitions,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        PartitionOwner partition_owner = round_robin
    );

    ~Shuffler();

    /**
     * @brief Shutdown the shuffle, blocking until all inflight communication is done.
     *
     * @throw cudf::logic_error If the shuffler is already inactive.
     */
    void shutdown();

  public:
    /**
     * @brief Insert a chunk into the shuffle.
     *
     * @param chunk The chunk to insert.
     */
    void insert(detail::Chunk&& chunk);

    /**
     * @brief Insert a packed (serialized) chunk into the shuffle.
     *
     * @param pid The partition ID the chunk belong to.
     * @param chunk The packed chunk, `cudf::table`, to insert.
     */
    void insert(PartID pid, cudf::packed_columns&& chunk);

    /**
     * @brief Insert a bunch of packed (serialized) chunks into the shuffle.
     *
     * @param chunks A map of partition IDs and their packed chunks.
     */
    void insert(std::unordered_map<PartID, cudf::packed_columns>&& chunks);

    /**
     * @brief Insert a finish mark for a partition.
     *
     * This tells the shuffler that no more chunks of the specified partition are coming.
     *
     * @param pid The partition ID to mark as finished.
     */
    void insert_finished(PartID pid);

    /**
     * @brief Extract all chunks of a specific partition.
     *
     * @param pid The partition ID.
     * @return A vector of packed columns (chunks) for the partition.
     */
    [[nodiscard]] std::vector<cudf::packed_columns> extract(PartID pid);

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
    /**
     * @brief Insert a chunk into the outbox (the chunk is ready for the user).
     *
     * @param chunk The chunk to insert.
     */
    void insert_into_outbox(detail::Chunk&& chunk);

    /**
     * @brief The event loop running by the shuffler's worker thread.
     *
     * @param self The shuffler instance.
     */
    static void event_loop(Shuffler* self);

    /**
     * @brief Executes a single iteration of the shuffler's event loop.
     *
     * This function manages the movement of data chunks between ranks in the distributed
     * system, handling tasks such as sending and receiving metadata, GPU data, and
     * readiness messages. It also manages the processing of chunks in transit, both
     * outgoing and incoming, and updates the necessary data structures for further
     * processing.
     *
     * @param self The `Shuffler` instance that owns the event loop.
     * @param fire_and_forget Ongoing "fire-and-forget" operations (non-blocking sends).
     * @param incoming_chunks Chunks ready to be received.
     * @param outgoing_chunks Chunks ready to be sent.
     * @param in_transit_chunks Chunks currently in transit.
     * @param in_transit_futures Futures corresponding to in-transit chunks.
     */
    static void run_event_loop_iteration(
        Shuffler& self,
        std::vector<std::unique_ptr<Communicator::Future>>& fire_and_forget,
        std::multimap<Rank, detail::Chunk>& incoming_chunks,
        std::unordered_map<detail::ChunkID, detail::Chunk>& outgoing_chunks,
        std::unordered_map<detail::ChunkID, detail::Chunk>& in_transit_chunks,
        std::unordered_map<detail::ChunkID, std::unique_ptr<Communicator::Future>>&
            in_transit_futures
    );

    /// @brief Get an new unique chunk ID.
    [[nodiscard]] detail::ChunkID get_new_cid();

    /**
     * @brief Create a new chunk from metadata and gpu data.
     *
     * The chunk is assigned a new unique ID using `get_new_cid()`.
     *
     * @param pid The partition ID of the new chunk.
     * @param metadata The metadata of the new chunk, can be null.
     * @param gpu_data The gpu data of the new chunk, can be null.
     */
    [[nodiscard]] detail::Chunk create_chunk(
        PartID pid,
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data
    ) {
        return detail::Chunk{
            pid,
            get_new_cid(),
            0,  // expected_num_chunks
            gpu_data ? gpu_data->size() : 0,  // gpu_data_size
            std::move(metadata),
            br_->move(std::move(gpu_data), stream_)
        };
    }

  public:
    PartID const total_num_partitions;  ///< Total number of partition in the shuffle.
    PartitionOwner const partition_owner;  ///< Function to determine partition ownership

  private:
    rmm::cuda_stream_view stream_;
    BufferResource* br_;
    bool active_{true};
    detail::PostBox inbox_;
    detail::PostBox outbox_;

    std::shared_ptr<Communicator> comm_;
    OpID const op_id_;
    std::thread event_loop_thread_;
    std::atomic<bool> event_loop_thread_run_{true};

    detail::FinishCounter finish_counter_;
    std::unordered_map<PartID, detail::ChunkID> outbound_chunk_counter_;
    mutable std::mutex outbound_chunk_counter_mutex_;

    std::atomic<detail::ChunkID> chunk_id_counter_{0};
};

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
