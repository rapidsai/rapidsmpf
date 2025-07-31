/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>


class ShuffleInsertGroupedTest;

/**
 * @namespace rapidsmpf::shuffler
 * @brief Shuffler interfaces.
 *
 * A shuffle service for host and device data. Use `Shuffler` to perform a single shuffle.
 */
namespace rapidsmpf::shuffler {

/**
 * @brief Shuffle service for cuDF tables.
 *
 * The `Shuffler` class provides an interface for performing a shuffle operation on cuDF
 * tables, using a partitioning scheme to distribute and collect data chunks across
 * different ranks.
 */
class Shuffler {
    friend class ::ShuffleInsertGroupedTest;

  public:
    /**
     * @brief Function that given a `Communicator` and a `PartID`, returns the
     * `rapidsmpf::Rank` of the _owning_ node.
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
        return static_cast<Rank>(pid % static_cast<PartID>(comm->nranks()));
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
     * @param progress_thread The progress thread to use.
     * @param op_id The operation ID of the shuffle. This ID is unique for this operation,
     * and should not be reused until all nodes has called `Shuffler::shutdown()`.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param stream The CUDA stream for memory operations.
     * @param br Buffer resource used to allocate temporary and the shuffle result.
     * @param statistics The statistics instance to use (disabled by default).
     * @param partition_owner Function to determine partition ownership.
     */
    Shuffler(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        OpID op_id,
        PartID total_num_partitions,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        PartitionOwner partition_owner = round_robin
    );

    ~Shuffler();

    /**
     * @brief Shutdown the shuffle, blocking until all inflight communication is done.
     *
     * @throw std::logic_error If the shuffler is already inactive.
     */
    void shutdown();

    /**
     * @brief Insert a chunk into the shuffle.
     *
     * @param chunk The chunk to insert.
     */
    void insert(detail::Chunk&& chunk);

    /**
     * @brief Insert a map of packed data, grouping them by destination rank, and
     * concatenating into a single chunk per rank.
     *
     * @param chunks A map of partition IDs and their packed chunks.
     */
    void concat_insert(std::unordered_map<PartID, PackedData>&& chunks);

    /**
     * @brief Insert a bunch of packed (serialized) chunks into the shuffle.
     *
     * @param chunks A map of partition IDs and their packed chunks.
     */
    void insert(std::unordered_map<PartID, PackedData>&& chunks);

    /**
     * @brief Insert a finish mark for a partition.
     *
     * This tells the shuffler that no more chunks of the specified partition are coming.
     *
     * @param pid The partition ID to mark as finished.
     */
    void insert_finished(PartID pid);

    /**
     * @brief Insert a finish mark for a list of partitions.
     *
     * @param pids The list of partition IDs to mark as finished.
     */
    void insert_finished(std::vector<PartID>&& pids);

    /**
     * @brief Extract all chunks of a specific partition.
     *
     * @param pid The partition ID.
     * @return A vector of packed data (chunks) for the partition.
     */
    [[nodiscard]] std::vector<PackedData> extract(PartID pid);

    /**
     * @brief Check if all partitions are finished.
     *
     * @return True if all partitions are finished, otherwise False.
     */
    [[nodiscard]] bool finished() const;

    /**
     * @brief Wait for any partition to finish.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @return The partition ID of the next finished partition.
     */
    PartID wait_any(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Wait for a specific partition to finish (blocking).
     *
     * @param pid The desired partition ID.
     * @param timeout Optional timeout (ms) to wait.
     */
    void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Wait for at least one partition to finish.
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @return The partition IDs of all finished partitions.
     */
    std::vector<PartID> wait_some(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Spills data to device if necessary.
     *
     * This function has two modes:
     *  - If `amount` is specified, it tries to spill at least `amount` bytes of
     *    device memory.
     *  - If `amount` is not specified (the default case), it spills based on the
     *    current available device memory returned by the buffer resource.
     *
     * In both modes, it adds to the "spill-device-limit-breach" statistic if not
     * enough memory could be spilled.
     *
     * @param amount An optional amount of memory to spill. If not provided, the
     * function will check the current available device memory.
     * @return The amount of memory actually spilled.
     */
    std::size_t spill(std::optional<std::size_t> amount = std::nullopt);

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
    void insert_into_ready_postbox(detail::Chunk&& chunk);

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
     * @param stream The CUDA stream for BufferResource memory operations.
     * @param event The event to use for the new chunk.
     */
    [[nodiscard]] detail::Chunk create_chunk(
        PartID pid, PackedData&& packed_data, std::shared_ptr<Buffer::Event> event
    );

  public:
    PartID const total_num_partitions;  ///< Total number of partition in the shuffle.
    PartitionOwner const partition_owner;  ///< Function to determine partition ownership

  private:
    rmm::cuda_stream_view stream_;
    BufferResource* br_;
    bool active_{true};
    detail::PostBox<Rank> outgoing_postbox_;  ///< Postbox for outgoing chunks, that are
                                              ///< ready to be sent to other ranks.
    detail::PostBox<PartID> ready_postbox_;  ///< Postbox for received chunks, that are
                                             ///< ready to be extracted by the user.

    std::shared_ptr<Communicator> comm_;
    std::shared_ptr<ProgressThread> progress_thread_;
    ProgressThread::FunctionID progress_thread_function_id_;
    OpID const op_id_;

    SpillManager::SpillFunctionID spill_function_id_;

    detail::FinishCounter finish_counter_;
    std::unordered_map<PartID, detail::ChunkID> outbound_chunk_counter_;
    mutable std::mutex outbound_chunk_counter_mutex_;

    // We protect ready_postbox extraction to avoid returning a chunk that is in the
    // process of being spilled by `Shuffler::spill`.
    mutable std::mutex ready_postbox_spilling_mutex_;

    std::atomic<detail::ChunkID> chunk_id_counter_{0};

    std::shared_ptr<Statistics> statistics_;

    class Progress;
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

}  // namespace rapidsmpf::shuffler
