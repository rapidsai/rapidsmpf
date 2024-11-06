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


#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/chunk.hpp>

namespace rapidsmp::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 */
class PostBox {
  public:
    /**
     * @brief Default constructor for PostBox.
     */
    PostBox() = default;

    /**
     * @brief Inserts a chunk into the PostBox.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk);

    /**
     * @brief Extracts a specific chunk from the PostBox.
     *
     * @param pid The ID of the partition.
     * @param cid The ID of the chunk.
     * @return The extracted chunk.
     *
     * @throws std::out_of_range If the specified chunk is not found.
     */
    Chunk extract(PartID pid, ChunkID cid);

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
     * @brief Checks if the PostBox is empty.
     *
     * @return `true` if the PostBox is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const {
        return pigeonhole_.empty();
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const;

  private:
    std::mutex mutex_;
    std::unordered_map<PartID, std::unordered_map<ChunkID, Chunk>>
        pigeonhole_;  ///< Storage for chunks, organized by partition and chunk ID.
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

}  // namespace rapidsmp::shuffler::detail
