/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

/**
 * @namespace rapidsmpf::shuffler
 * @brief Shuffler interfaces.
 *
 * A shuffle service for host and device data. Use `Shuffler` to perform a single shuffle.
 */
namespace rapidsmpf::shuffler {

/**
 * @brief Shuffle service for all-to-all style communication of partitioned data.
 *
 * The `Shuffler` class provides an interface for performing a shuffle operation on
 * distributed data, using a partitioning scheme to distribute and collect data chunks
 * across different ranks.
 */
class Shuffler {
  public:
    /**
     * @brief Function that given a `Communicator`, `PartID`, and total partition count,
     * returns the `rapidsmpf::Rank` of the _owning_ node.
     */
    using PartitionOwner =
        std::function<Rank(std::shared_ptr<Communicator> const&, PartID, PartID)>;

    /**
     * @brief A `PartitionOwner` that distributes partitions using round robin assignment.
     *
     * @param comm The communicator to use.
     * @param pid The partition ID to query.
     * @param total_num_partitions Total number of partitions (unused).
     * @return The rank owning the partition.
     */
    static Rank round_robin(
        std::shared_ptr<Communicator> const& comm,
        PartID pid,
        [[maybe_unused]] PartID total_num_partitions
    ) {
        return safe_cast<Rank>(pid % safe_cast<PartID>(comm->nranks()));
    }

    /**
     * @brief A `PartitionOwner` that assigns contiguous partition ID ranges to ranks.
     *
     * Rank 0 gets [0, k), rank 1 gets [k, 2k), etc. Use for sort so that each rank's
     * local_partitions() are adjacent and in order.
     *
     * @param comm The communicator to use.
     * @param pid The partition ID to query.
     * @param total_num_partitions Total number of partitions (must match the shuffle).
     * @return The rank owning the partition.
     */
    static Rank contiguous(
        std::shared_ptr<Communicator> const& comm, PartID pid, PartID total_num_partitions
    ) {
        return safe_cast<Rank>(
            (pid * safe_cast<PartID>(comm->nranks())) / total_num_partitions
        );
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
     * @brief Callback function type called when all partitions are finished and data
     * can be extracted.
     *
     * @warning A callback must be fast and non-blocking. Ideally it should be used
     * to signal a separate thread to do the actual processing.
     */
    using FinishedCallback = std::function<void()>;

    /**
     * @brief Construct a new shuffler for a single shuffle.
     *
     * @param comm The communicator to use.
     * @param op_id The operation ID of the shuffle.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param br Buffer resource used to allocate temporary and the shuffle result.
     * @param finished_callback Callback to notify when all partitions are finished.
     * @param partition_owner Function to determine partition ownership.
     * @param mpe Optional custom metadata payload exchange. If not provided,
     * uses the default tag-based implementation.
     *
     * @note It is safe to reuse the `op_id` as soon as `wait` has completed
     * locally.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     */
    Shuffler(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        PartID total_num_partitions,
        BufferResource* br,
        FinishedCallback&& finished_callback,
        PartitionOwner partition_owner = round_robin,
        std::unique_ptr<communicator::MetadataPayloadExchange> mpe = nullptr
    );

    /**
     * @brief Construct a new shuffler for a single shuffle.
     *
     * @param comm The communicator to use.
     * @param op_id The operation ID of the shuffle. This ID is unique for this operation,
     * and should not be reused until all nodes has called `Shuffler::shutdown()`.
     * @param total_num_partitions Total number of partitions in the shuffle.
     * @param br Buffer resource used to allocate temporary and the shuffle result.
     * @param partition_owner Function to determine partition ownership.
     * @param mpe Optional custom metadata payload exchange. If not provided,
     * uses the default tag-based implementation.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     */
    Shuffler(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        PartID total_num_partitions,
        BufferResource* br,
        PartitionOwner partition_owner = round_robin,
        std::unique_ptr<communicator::MetadataPayloadExchange> mpe = nullptr
    )
        : Shuffler(
              comm,
              op_id,
              total_num_partitions,
              br,
              nullptr,
              partition_owner,
              std::move(mpe)
          ) {}

    ~Shuffler();

    /**
     * @brief Gets the communicator associated with this Shuffler.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept {
        return comm_;
    }

    Shuffler(Shuffler const&) = delete;
    Shuffler& operator=(Shuffler const&) = delete;

    /**
     * @brief Shutdown the shuffle, blocking until all inflight communication is done.
     *
     * @throws std::logic_error If the shuffler is already inactive.
     */
    void shutdown();

    /**
     * @brief Insert a bunch of packed (serialized) chunks into the shuffle.
     *
     * @note Concurrent insertion by multiple threads is supported, the caller must ensure
     * that `insert_finished()` is called _after_ all `insert()` calls have completed.
     *
     * @param chunks A map of partition IDs and their packed chunks.
     */
    void insert(std::unordered_map<PartID, PackedData>&& chunks);

    /**
     * @brief Signal that no more data will be inserted into the shuffle.
     *
     * This informs the shuffler that this rank has finished inserting data. Must be
     * called exactly once.
     *
     * @note If multiple threads are `insert()`ing, you must establish a happens-before
     * relationship between the completion of all `insert()`s and the final call to
     * `insert_finished()`.
     */
    void insert_finished();

    /**
     * @brief Extract all chunks belonging to the specified partition.
     *
     * It is valid to extract a partition that has not yet been fully received.
     * In such cases, only the chunks received so far are returned.
     *
     * To ensure the partition is complete, use `wait()`
     * or another appropriate synchronization mechanism beforehand.
     *
     * @param pid The ID of the partition to extract.
     * @return A vector of PackedData chunks associated with the partition.
     */
    [[nodiscard]] std::vector<PackedData> extract(PartID pid);

    /**
     * @brief Check if all partitions are finished.
     *
     * @return True if all partitions are finished, otherwise False.
     */
    [[nodiscard]] bool finished() const;

    /**
     * @brief Wait for all partitions to finish (blocking).
     *
     * @param timeout Optional timeout (ms) to wait.
     *
     * @throws std::runtime_error if the timeout is reached.
     */
    void wait(std::optional<std::chrono::milliseconds> timeout = {});

    /**
     * @brief Spills data to device if necessary.
     *
     * This function has two modes:
     *  - If `amount` is specified, it tries to spill at least `amount` bytes of
     *    device memory.
     *  - If `amount` is not specified (the default case), it spills based on the
     *    current available device memory returned by the buffer resource.
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

    /**
     * @brief Returns the local partition IDs owned by the shuffler`.
     *
     * @return A span of partition IDs owned by the shuffler.
     */
    [[nodiscard]] std::span<PartID const> local_partitions() const;

    /**
     * @brief The number of bits used to store the counter in a chunk ID.
     */
    static constexpr int chunk_id_counter_bits = 38;

    /**
     * @brief Extract the rank from a chunk ID.
     * @param cid The chunk ID.
     * @return The rank.
     */
    static constexpr Rank extract_rank(detail::ChunkID cid) {
        return safe_cast<Rank>(cid >> chunk_id_counter_bits);
    }

  private:
    /**
     * @brief Insert a chunk into the shuffle.
     *
     * @param chunk The chunk to insert.
     */
    void insert(detail::Chunk&& chunk);

    /**
     * @brief Insert a chunk into the received box (the chunk is ready for the user).
     *
     * @param chunk The chunk to insert.
     */
    void insert_into_received(detail::Chunk&& chunk);

    /// @brief Get an new unique chunk ID.
    [[nodiscard]] detail::ChunkID get_new_cid();

    /**
     * @brief Create a new chunk from metadata and GPU data.
     *
     * The chunk is assigned a new unique ID using `get_new_cid()`.
     *
     * @param pid The partition ID of the new chunk.
     * @param packed_data The pack data of the new chunk.
     */
    [[nodiscard]] detail::Chunk create_chunk(PartID pid, PackedData&& packed_data);

  public:
    PartID const total_num_partitions;  ///< Total number of partition in the shuffle.
    PartitionOwner const partition_owner;  ///< Function to determine partition ownership

  private:
    BufferResource* br_;
    std::atomic<bool> active_{true};
    // Have we called `insert_finished()` on this rank.
    std::atomic<bool> locally_finished_{false};
    // Flipped to true exactly once when partitions are ready for extraction and we've
    // posted all sends we're going to
    bool can_extract_{false};
    detail::ChunksToSend to_send_;  ///< Storage for chunks to send to other ranks.
    detail::ReceivedChunks received_;  ///< Storage for received chunks that are
                                       ///< ready to be extracted by the user.

    std::shared_ptr<Communicator> comm_;
    std::unique_ptr<communicator::MetadataPayloadExchange> mpe_;
    ProgressThread::FunctionID progress_thread_function_id_;

    SpillManager::SpillFunctionID spill_function_id_;

    std::vector<PartID> const local_partitions_;

    detail::FinishCounter finish_counter_;
    std::vector<detail::ChunkID> outbound_chunk_counter_;  ///< indexed by Rank
    std::atomic<detail::ChunkID> chunk_id_counter_{0};

    std::shared_ptr<Statistics> statistics_;

    // For notifications.
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    FinishedCallback finished_callback_;  ///< Called once when data can be extracted.

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
