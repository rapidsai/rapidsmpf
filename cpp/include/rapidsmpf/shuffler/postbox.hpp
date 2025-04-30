/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once


#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 */
class PostBox {
  public:
    /**
     * @brief Constructor for PostBox.
     *
     * @param nranks The number of ranks in the communicator.
     * @param partition_owner A function that maps a partition ID to a rank.
     */
    PostBox(Rank nranks, std::function<Rank(PartID)>&& partition_owner);

    /**
     * @brief Inserts a chunk into the PostBox.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk);

    /**
     * @brief Extracts a specific chunk from the PostBox.
     *
     * @param pid The ID of the partition containing the chunk.
     * @param cid The ID of the chunk to be accessed.
     * @return The extracted chunk.
     *
     * @throws std::out_of_range If the chunk is not found.
     */
    [[nodiscard]] Chunk extract(PartID pid, ChunkID cid);

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A map of chunk IDs to chunks for the specified partition.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract(PartID pid);

    /**
     * @brief Extracts all chunks from the PostBox.
     *
     * @return A vector of all chunks in the PostBox.
     */
    std::vector<Chunk> extract_all();

    /**
     * @brief Extracts all chunks from the PostBox for a specific rank.
     *
     * @param rank The rank to extract chunks for. If not provided, chunks from the first
     * available rank are returned.
     * @return A vector of all chunks in the PostBox for the specified rank. If the
     * PostBox is empty or the rank is not found, an empty vector is returned.
     */
    std::list<Chunk> extract_for_rank(std::optional<Rank> rank = std::nullopt) noexcept;

    /**
     * @brief Checks if the PostBox is empty.
     *
     * @return `true` if the PostBox is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief Searches for chunks of the specified memory type.
     *
     * @param mem_type The type of memory to search within.
     * @return A vector of tuples, where each tuple contains: PartID, ChunkID, and the
     * size of the chunk.
     */
    [[nodiscard]] std::vector<std::tuple<PartID, ChunkID, std::size_t>> search(
        MemoryType mem_type
    ) const;

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;

    // Note: list allows constant time insertion, removals, and concatenations (splice).
    std::unordered_map<Rank, std::list<Chunk>>
        pigeonhole_;  ///< Storage for chunks, organized by destination rank.

    // We may be able to remove the following members once we move to a rank-based
    // extraction.
    std::function<Rank(PartID)>
        partition_owner_;  ///< Function to determine partition owner.
    std::unordered_set<ChunkID> chunk_ids_;  ///< Set of chunk IDs in the PostBox.
};

/**
 * @brief Overloads the stream insertion operator for the PostBox class.
 *
 * This function allows a description of a PostBox to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, PostBox const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
