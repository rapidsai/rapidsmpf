/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {

template <typename KeyType>
void PostBox<KeyType>::insert(Chunk&& chunk) {
    std::lock_guard const lock(mutex_);
    auto [_, inserted] =
        pigeonhole_[key_map_fn_(chunk.pid)].insert({chunk.cid, std::move(chunk)});
    RAPIDSMPF_EXPECTS(inserted, "PostBox.insert(): chunk already exist");
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
std::vector<Chunk> PostBox<KeyType>::extract_all() {
    std::lock_guard const lock(mutex_);
    std::vector<Chunk> ret;
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& [_, chunk] : chunks) {
            ret.push_back(std::move(chunk));
        }
    }
    pigeonhole_.clear();
    return ret;
}

template <typename KeyType>
bool PostBox<KeyType>::empty() const {
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
            if (chunk.gpu_data && chunk.gpu_data->mem_type() == mem_type) {
                ret.emplace_back(key, cid, chunk.gpu_data->size);
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
            assert(cid == chunk.cid);
            if (chunk.expected_num_chunks) {
                ss << "EOP" << chunk.expected_num_chunks << ", ";
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
