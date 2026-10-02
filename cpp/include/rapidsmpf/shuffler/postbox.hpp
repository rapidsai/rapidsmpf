/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <mutex>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief Policy deciding which outgoing chunks may overtake a chunk that cannot be sent
 * yet (because it is not ready or its disk-backed data cannot be restored).
 *
 * Insertion order per partition is restored on the receiving side by sorting on chunk
 * ID, so these policies only bound how much data is in flight, not correctness.
 */
enum class SendOrderPolicy : std::uint8_t {
    Global,  ///< A blocked chunk blocks every later chunk.
    PerRank,  ///< A blocked chunk blocks later chunks to the same destination rank.
    PerPartition,  ///< A blocked chunk blocks later chunks of the same partition.
    None,  ///< A blocked chunk never blocks other chunks.
};

/**
 * @brief Parse a send order policy name.
 *
 * @param name One of "global", "rank", "pid" or "none" (case-insensitive).
 * @return The parsed policy.
 *
 * @throws std::invalid_argument If @p name is not a valid policy name.
 */
SendOrderPolicy parse_send_order_policy(std::string_view name);

/**
 * @brief Read the send order policy from `RAPIDSMPF_SHUFFLER_SEND_ORDER_POLICY`.
 *
 * TODO: this is a temporary backdoor for benchmarking send order policies.
 *
 * @return The policy in the environment variable, or `SendOrderPolicy::Global` if unset.
 */
SendOrderPolicy send_order_policy_from_env();

/**
 * @brief Overloads the stream insertion operator for SendOrderPolicy.
 *
 * @param os The output stream to write to.
 * @param policy The policy to write.
 * @return A reference to the modified output stream.
 */
std::ostream& operator<<(std::ostream& os, SendOrderPolicy policy);

/**
 * @brief A thread-safe container for managing outgoing (to send) chunks.
 */
class ChunksToSend {
  public:
    /**
     * @brief Counters describing a single extraction.
     */
    struct ExtractStats {
        std::size_t not_ready{0};  ///< Chunks skipped because they were not ready.
        std::size_t blocked{0};  ///< Chunks held back behind a blocked chunk.
        std::size_t restored{0};  ///< Disk-backed chunks restored.
        std::size_t restore_deferred{0};  ///< Disk-backed chunks left on disk.
    };

    /**
     * @brief Construct a new container.
     *
     * @param policy The policy deciding which chunks may overtake a blocked chunk.
     */
    ChunksToSend(SendOrderPolicy policy = SendOrderPolicy::Global) : policy_{policy} {}

    /**
     * @brief Insert a chunk into the container.
     *
     * @param dst The destination rank of the chunk.
     * @param c The chunk to insert.
     */
    void insert(Rank dst, std::unique_ptr<Chunk> c);

    /**
     * @brief Extract ready chunks.
     *
     * Disk-backed chunks are extracted as-is, without restoring them.
     *
     * @note Ready means no stream-ordered work queued on the chunk's data.
     *
     * @param stats If not null, filled with counters describing this extraction.
     * @return Vector of chunks ready to send.
     */
    [[nodiscard]] std::vector<Chunk> extract_ready(ExtractStats* stats = nullptr);

    /**
     * @brief Extract ready chunks and restore disk-backed data to addressable memory.
     *
     * At most one disk-backed chunk is restored per call. A chunk that is not ready, or
     * a disk-backed chunk that is not restored, blocks later chunks as decided by the
     * send order policy. With `SendOrderPolicy::Global`, extraction stops at the first
     * such chunk and a failed restore throws.
     *
     * @param br The buffer resource used to restore disk-backed data.
     * @param memory_types Addressable memory types to try in preference order.
     * @param stats If not null, filled with counters describing this extraction.
     * @return Vector of chunks ready to send.
     *
     * @throws std::runtime_error If the policy is `SendOrderPolicy::Global` and no
     * memory can be reserved to restore a disk-backed chunk.
     */
    [[nodiscard]] std::vector<Chunk> extract_and_restore(
        BufferResource* br,
        std::span<MemoryType const> memory_types,
        ExtractStats* stats = nullptr
    );

    /**
     * @brief @return The send order policy.
     */
    [[nodiscard]] SendOrderPolicy policy() const noexcept {
        return policy_;
    }

    /**
     * @brief @return Whether the container is empty.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return Returns a description of this instance.
     */
    [[nodiscard]] std::string str() const;

  private:
    [[nodiscard]] std::vector<Chunk> extract(
        BufferResource* br, std::span<MemoryType const> memory_types, ExtractStats* stats
    );

    SendOrderPolicy const policy_;
    mutable std::mutex mutex_{};
    std::vector<std::pair<Rank, std::unique_ptr<Chunk>>> chunks_{};
};

/**
 * @brief Overloads the stream insertion operator for the ChunksToSend class.
 *
 * This function allows a description of ChunksToSend to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ChunksToSend const& obj) {
    os << obj.str();
    return os;
}

/**
 * @brief A thread-safe container for managing received chunks stratified by partition ID.
 */
class ReceivedChunks {
  public:
    /**
     * @brief Construct a new container.
     *
     * @param num_keys_hint The number of keys to reserve space for.
     */
    ReceivedChunks(std::size_t num_keys_hint = 0) {
        if (num_keys_hint > 0) {
            pigeonhole_.reserve(num_keys_hint);
        }
    }

    /**
     * @brief Insert a chunk.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk);

    /**
     * @brief Check whether the specified partition contains any chunks.
     *
     * @param pid Identifier of the partition to query.
     * @return True if the partition contains no chunks, false otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool is_empty(PartID pid) const;

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A vector of chunks.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    [[nodiscard]] std::vector<Chunk> extract(PartID pid);

    /**
     * @brief Checks if the container is empty.
     *
     * @return `true` if the container is empty, `false` otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return A description of this container.
     */
    [[nodiscard]] std::string str() const;

    /**
     * @brief Spill received device payloads.
     *
     * Moves device buffers to the first immediately available destination in
     * @p spillable_memory_types.
     *
     * @param br The buffer resource for memory and disk allocations.
     * @param amount Requested amount of device data to spill in bytes.
     * @param spillable_memory_types Non-device spill destinations in preference order.
     * @return Actual amount of device data spilled in bytes.
     */
    [[nodiscard]] std::size_t spill(
        BufferResource* br,
        std::size_t amount,
        std::span<MemoryType const> spillable_memory_types
    );

  private:
    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;
    std::unordered_map<PartID, std::vector<Chunk>>
        pigeonhole_;  ///< Storage for chunks, stratified by partition ID.
};

/**
 * @brief Overloads the stream insertion operator for the ReceivedChunks class.
 *
 * This function allows a description of ReceivedChunks be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ReceivedChunks const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
