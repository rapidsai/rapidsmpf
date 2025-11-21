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
    // check if all partition IDs in the chunk map to the same key
    KeyType key = key_map_fn_(chunk.part_id(0));
    for (size_t i = 1; i < chunk.n_messages(); ++i) {
        RAPIDSMPF_EXPECTS(
            key == key_map_fn_(chunk.part_id(i)),
            "PostBox.insert(): all messages in the chunk must map to the same key"
        );
    }
    // std::lock_guard const lock(mutex_);
    auto& map_value = pigeonhole_.at(key);
    std::lock_guard lock(map_value.mutex);
    if (map_value.chunks.empty()) {
        RAPIDSMPF_EXPECTS(
            n_non_empty_keys_.fetch_add(1, std::memory_order_relaxed) + 1
                <= pigeonhole_.size(),
            "PostBox.insert(): n_non_empty_keys_ is already at the maximum"
        );
    }
    RAPIDSMPF_EXPECTS(
        map_value.chunks.emplace(std::move(chunk)).second,
        "PostBox.insert(): chunk already exist"
    );
}

template <typename KeyType>
bool PostBox<KeyType>::is_empty(PartID pid) const {
    auto& map_value = pigeonhole_.at(key_map_fn_(pid));
    std::lock_guard lock(map_value.mutex);
    return map_value.chunks.empty();
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract(PartID pid) {
    return extract_by_key(key_map_fn_(pid));
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract_by_key(KeyType key) {
    auto& map_value = pigeonhole_.at(key);
    std::vector<Chunk> ret;
    std::lock_guard lock(map_value.mutex);
    RAPIDSMPF_EXPECTS(!map_value.chunks.empty(), "PostBox.extract(): partition is empty");
    ret.reserve(map_value.chunks.size());

    for (auto it = map_value.chunks.begin(); it != map_value.chunks.end();) {
        auto node = map_value.chunks.extract(it++);
        ret.emplace_back(std::move(node.value()));
    }

    RAPIDSMPF_EXPECTS(
        n_non_empty_keys_.fetch_sub(1, std::memory_order_relaxed) > 0,
        "PostBox.extract(): n_non_empty_keys_ is already 0"
    );
    return ret;
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract_all_ready() {
    std::vector<Chunk> ret;

    // Iterate through the outer map
    for (auto& [key, map_value] : pigeonhole_) {
        std::lock_guard lock(map_value.mutex);
        bool chunks_available = !map_value.chunks.empty();
        auto chunk_it = map_value.chunks.begin();
        while (chunk_it != map_value.chunks.end()) {
            if (chunk_it->is_ready()) {
                auto node = map_value.chunks.extract(chunk_it++);
                ret.emplace_back(std::move(node.value()));
            } else {
                ++chunk_it;
            }
        }

        // if the chunks were available and are now empty, its fully extracted
        if (chunks_available && map_value.chunks.empty()) {
            RAPIDSMPF_EXPECTS(
                n_non_empty_keys_.fetch_sub(1, std::memory_order_relaxed) > 0,
                "PostBox.extract_all_ready(): n_non_empty_keys_ is already 0"
            );
        }
    }
    return ret;
}

template <typename KeyType>
bool PostBox<KeyType>::empty() const {
    return n_non_empty_keys_.load(std::memory_order_acquire) == 0;
}

template <typename KeyType>
size_t PostBox<KeyType>::spill(BufferResource* /* br */, size_t /* amount */) {
    // TODO: implement spill
    return 0;
}

template <typename KeyType>
std::string PostBox<KeyType>::str() const {
    if (empty()) {
        return "PostBox()";
    }
    std::stringstream ss;
    ss << "PostBox(";
    for (auto const& [key, map_value] : pigeonhole_) {
        ss << "k=" << key << ": [";
        for (auto const& chunk : map_value.chunks) {
            // assert(cid == chunk.chunk_id());
            if (chunk.is_control_message(0)) {
                ss << "EOP" << chunk.expected_num_chunks(0) << ", ";
            } else {
                ss << chunk.chunk_id() << ", ";
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
