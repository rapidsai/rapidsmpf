/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 *
 * @tparam KeyType The type of the key used to map chunks.
 */
template <typename KeyType>
class PostBox {
  public:
    using key_type = KeyType;  ///< The type of the key used to map chunks.

    /**
     * @brief Construct a new PostBox.
     *
     * @tparam Fn The type of the function that maps a partition ID to a key.
     * @param key_map_fn A function that maps a partition ID to a key.
     * @param keys The keys expected to be used in the PostBox.
     *
     * @note The `key_map_fn` must be convertible to a function that takes a `PartID` and
     * returns a `KeyType`.
     */
    template <typename Fn, std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, KeyType>
    PostBox(Fn&& key_map_fn, Range&& keys) : key_map_fn_(std::move(key_map_fn)) {
        pigeonhole_.reserve(std::ranges::size(keys));
        for (const auto& key : keys) {
            pigeonhole_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(key),
                std::forward_as_tuple()
            );
        }
    }

    /**
     * @brief Inserts a chunk into the PostBox.
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
     * @return A map of chunk IDs to chunks for the specified partition.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    std::vector<Chunk> extract(PartID pid);

    /**
     * @brief Extracts all chunks associated with a specific key.
     *
     * @param key The key.
     * @return A map of chunk IDs to chunks for the specified key.
     *
     * @throws std::out_of_range If the key is not found.
     */
    std::vector<Chunk> extract_by_key(KeyType key);

    /**
     * @brief Extracts all ready chunks from the PostBox.
     *
     * @return A vector of all ready chunks in the PostBox.
     */
    std::vector<Chunk> extract_all_ready();

    /**
     * @brief Checks if the PostBox is empty.
     *
     * @return `true` if the PostBox is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

    /**
     * @brief Spills the specified amount of data from the PostBox.
     *
     * @param br Buffer resource to use for spilling.
     * @param log Logger to use for logging.
     * @param amount The amount of data to spill.
     * @return The amount of data spilled.
     */
    size_t spill(BufferResource* br, Communicator::Logger& log, size_t amount);

  private:
    /**
     * @brief Map value for the PostBox.
     */
    struct MapValue {
        mutable std::mutex mutex;  ///< Mutex to protect each key
        std::vector<Chunk> chunks;  ///< Vector of chunks for the key
    };

    std::function<key_type(PartID)>
        key_map_fn_;  ///< Function to map partition IDs to keys.
    std::unordered_map<key_type, MapValue> pigeonhole_;  ///< Storage for chunks
    std::atomic<size_t> n_non_empty_keys_{0};
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
template <typename KeyType>
inline std::ostream& operator<<(std::ostream& os, PostBox<KeyType> const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
