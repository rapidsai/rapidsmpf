/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 *
 * @tparam KeyType The type of the key used to map chunks.
 */
template <typename KeyType>
class PostBox2 {
  public:
    using key_type = KeyType;  ///< The type of the key used to map chunks.

    /**
     * @brief Construct a new PostBox2.
     *
     * @tparam Fn The type of the function that maps a partition ID to a key.
     * @param key_map_fn A function that maps a partition ID to a key.
     * @param num_keys_hint The number of keys to reserve space for.
     *
     * @note The `key_map_fn` must be convertible to a function that takes a `PartID` and
     * returns a `KeyType`.
     */
    template <typename Fn>
    PostBox2(Fn&& key_map_fn, size_t num_keys_hint = 0)
        : key_map_fn_(std::move(key_map_fn)) {
        if (num_keys_hint > 0) {
            for (auto& bucket : pigeonhole_buckets_) {
                bucket.pigeonholes.reserve(num_keys_hint / kNumBuckets);
            }
        }
    }

    /**
     * @brief Inserts a chunk into the PostBox2.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk) {
        // check if all partition IDs in the chunk map to the same key
        KeyType key = key_map_fn_(chunk.part_id(0));
        for (size_t i = 1; i < chunk.n_messages(); ++i) {
            RAPIDSMPF_EXPECTS(
                key == key_map_fn_(chunk.part_id(i)),
                "PostBox2.insert(): all messages in the chunk must map to the same key"
            );
        }
        auto& bucket = pigeonhole_buckets_[bucket_index(key)];
        std::lock_guard const lock(bucket.mtx);
        auto [_, inserted] =
            bucket.pigeonholes[key].emplace(chunk.chunk_id(), std::move(chunk));
        RAPIDSMPF_EXPECTS(inserted, "PostBox.insert(): chunk already exist");
    }

    /**
     * @brief Extracts a specific chunk from the PostBox2.
     *
     * @param pid The ID of the partition containing the chunk.
     * @param cid The ID of the chunk to be accessed.
     * @return The extracted chunk.
     *
     * @throws std::out_of_range If the chunk is not found.
     */
    [[nodiscard]] Chunk extract(PartID pid, ChunkID cid) {
        KeyType key = key_map_fn_(pid);
        auto& bucket = pigeonhole_buckets_[bucket_index(key)];
        std::lock_guard const lock(bucket.mtx);
        return extract_item(bucket.pigeonholes[key], cid).second;
    }

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A map of chunk IDs to chunks for the specified partition.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract(PartID pid) {
        KeyType key = key_map_fn_(pid);
        auto& bucket = pigeonhole_buckets_[bucket_index(key)];
        std::lock_guard const lock(bucket.mtx);
        return extract_value(bucket.pigeonholes, key);
    }

    /**
     * @brief Extracts all chunks associated with a specific key.
     *
     * @param key The key.
     * @return A map of chunk IDs to chunks for the specified key.
     *
     * @throws std::out_of_range If the key is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract_by_key(KeyType key) {
        auto& bucket = pigeonhole_buckets_[bucket_index(key)];
        std::lock_guard const lock(bucket.mtx);
        return extract_value(bucket.pigeonholes, key);
    }

    /**
     * @brief Extracts all ready chunks from the PostBox2.
     *
     * @return A vector of all ready chunks in the PostBox2.
     */
    std::vector<Chunk> extract_all_ready() {
        std::vector<Chunk> ret;

        for (auto& bucket : pigeonhole_buckets_) {
            // std::lock_guard const lock(bucket.mtx);
            std::unique_lock lock(bucket.mtx, std::defer_lock);
            if (!lock.try_lock()) {
                continue;
            }

            // Iterate through the outer map
            auto pid_it = bucket.pigeonholes.begin();
            while (pid_it != bucket.pigeonholes.end()) {
                // Iterate through the inner map
                auto& chunks = pid_it->second;
                auto chunk_it = chunks.begin();
                while (chunk_it != chunks.end()) {
                    if (chunk_it->second.is_ready()) {
                        ret.push_back(std::move(chunk_it->second));
                        chunk_it = chunks.erase(chunk_it);
                    } else {
                        ++chunk_it;
                    }
                }

                // Remove the pid entry if its chunks map is empty
                if (chunks.empty()) {
                    pid_it = bucket.pigeonholes.erase(pid_it);
                } else {
                    ++pid_it;
                }
            }
        }

        return ret;
    }

    /**
     * @brief Checks if the PostBox2 is empty.
     *
     * @return `true` if the PostBox2 is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const {
        for (auto& bucket : pigeonhole_buckets_) {
            std::lock_guard const lock(bucket.mtx);
            if (!bucket.pigeonholes.empty()) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Searches for chunks of the specified memory type.
     *
     * @param mem_type The type of memory to search within.
     * @return A vector of tuples, where each tuple contains: PartID, ChunkID, and the
     * size of the chunk.
     */
    [[nodiscard]] std::vector<std::tuple<key_type, ChunkID, std::size_t>> search(
        MemoryType mem_type
    ) const {
        std::vector<std::tuple<KeyType, ChunkID, std::size_t>> ret;

        for (auto& bucket : pigeonhole_buckets_) {
            std::lock_guard const lock(bucket.mtx);
            for (auto& [key, chunks] : bucket.pigeonholes) {
                for (auto& [cid, chunk] : chunks) {
                    if (!chunk.is_control_message(0)
                        && chunk.data_memory_type() == mem_type)
                    {
                        ret.emplace_back(key, cid, chunk.concat_data_size());
                    }
                }
            }
        }
        return ret;
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const {
        if (empty()) {
            return "PostBox()";
        }
        std::stringstream ss;
        ss << "PostBox(";
        for (auto& bucket : pigeonhole_buckets_) {
            std::lock_guard const lock(bucket.mtx);
            for (auto const& [key, chunks] : bucket.pigeonholes) {
                ss << "k=" << key << ": [";
                for (auto const& [cid, chunk] : chunks) {
                    assert(cid == chunk.chunk_id());
                    if (chunk.is_control_message(0)) {
                        ss << "EOP" << chunk.expected_num_chunks(0) << ", ";
                    } else {
                        ss << cid << ", ";
                    }
                }
                ss << "\b\b], ";
            }
        }
        ss << "\b\b)";
        return ss.str();
    }

  private:
    static constexpr size_t kNumBuckets = 16;  ///< Number of buckets.

    /// Get the index of the bucket for a given key.
    constexpr inline size_t bucket_index(KeyType key) {
        return static_cast<size_t>(key) % kNumBuckets;
    }

    /// A bucket of pigeonholes, that is protected by a mutex.
    struct PigeonholesBucket {
        std::unordered_map<KeyType, std::unordered_map<ChunkID, Chunk>> pigeonholes;
        std::mutex mtx;
    };

    mutable std::array<PigeonholesBucket, kNumBuckets> pigeonhole_buckets_{
    };  ///< Array of buckets, each protected by a mutex.

    std::function<key_type(PartID)>
        key_map_fn_;  ///< Function to map partition IDs to keys.
};

/**
 * @brief Overloads the stream insertion operator for the PostBox2 class.
 *
 * This function allows a description of a PostBox2 to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
template <typename KeyType>
inline std::ostream& operator<<(std::ostream& os, PostBox2<KeyType> const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail