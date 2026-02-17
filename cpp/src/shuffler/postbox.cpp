/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler::detail {

template <typename KeyType>
void PostBox<KeyType>::insert(Chunk&& chunk) {
    // Single-message chunks only - get the key for the one partition
    KeyType key = key_map_fn_(chunk.part_id());
    std::lock_guard const lock(mutex_);
    RAPIDSMPF_EXPECTS(
        pigeonhole_[key].emplace(chunk.chunk_id(), std::move(chunk)).second,
        "PostBox.insert(): chunk already exist"
    );
}

template <typename KeyType>
bool PostBox<KeyType>::is_empty(PartID pid) const {
    std::lock_guard const lock(mutex_);
    return !pigeonhole_.contains(key_map_fn_(pid));
}

template <typename KeyType>
Chunk PostBox<KeyType>::extract(PartID pid, ChunkID cid) {
    std::lock_guard const lock(mutex_);
    return extract_item(pigeonhole_[key_map_fn_(pid)], cid).second;
}

template <typename KeyType>
std::unordered_map<ChunkID, Chunk> PostBox<KeyType>::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, key_map_fn_(pid));
}

template <typename KeyType>
std::unordered_map<ChunkID, Chunk> PostBox<KeyType>::extract_by_key(KeyType key) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, key);
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract_all_ready() {
    std::lock_guard const lock(mutex_);
    std::vector<Chunk> ret;

    // Iterate through the outer map
    auto pid_it = pigeonhole_.begin();
    while (pid_it != pigeonhole_.end()) {
        // Iterate through the inner map
        auto& chunks = pid_it->second;
        auto chunk_it = chunks.begin();
        while (chunk_it != chunks.end()) {
            if (chunk_it->second.is_ready()) {
                ret.emplace_back(std::move(chunk_it->second));
                chunk_it = chunks.erase(chunk_it);
            } else {
                ++chunk_it;
            }
        }

        // Remove the pid entry if its chunks map is empty
        if (chunks.empty()) {
            pid_it = pigeonhole_.erase(pid_it);
        } else {
            ++pid_it;
        }
    }

    return ret;
}

template <typename KeyType>
bool PostBox<KeyType>::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

template <typename KeyType>
std::vector<std::tuple<KeyType, ChunkID, std::size_t>> PostBox<KeyType>::search(
    MemoryType mem_type
) const {
    std::lock_guard const lock(mutex_);
    std::vector<std::tuple<KeyType, ChunkID, std::size_t>> ret;
    for (auto& [key, chunks] : pigeonhole_) {
        for (auto& [cid, chunk] : chunks) {
            if (!chunk.is_control_message() && chunk.data_memory_type() == mem_type) {
                ret.emplace_back(key, cid, chunk.data_size());
            }
        }
    }
    return ret;
}

template <typename KeyType>
std::string PostBox<KeyType>::str() const {
    if (empty()) {
        return "PostBox()";
    }
    std::stringstream ss;
    ss << "PostBox(";
    for (auto const& [key, chunks] : pigeonhole_) {
        ss << "k=" << key << ": [";
        for (auto const& [cid, chunk] : chunks) {
            assert(cid == chunk.chunk_id());
            if (chunk.is_control_message()) {
                ss << "EOP" << chunk.expected_num_chunks() << ", ";
            } else {
                ss << cid << ", ";
            }
        }
        ss << "\b\b], ";
    }
    ss << "\b\b)";
    return ss.str();
}

// Explicit instantiation for PartID and Rank
template class PostBox<PartID>;
template class PostBox<Rank>;

}  // namespace rapidsmpf::shuffler::detail
