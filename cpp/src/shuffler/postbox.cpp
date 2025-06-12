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
    std::lock_guard const lock(mutex_);
    auto [_, inserted] = pigeonhole_[key].insert({chunk.chunk_id(), std::move(chunk)});
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
                ret.push_back(std::move(chunk_it->second));
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
std::vector<Chunk> PostBox<KeyType>::extract_all_ready_concat(
    size_t max_concat_size,
    std::function<ChunkID()> chunk_id_gen,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    std::lock_guard const lock(mutex_);
    std::vector<Chunk> ret;
    ret.reserve(pigeonhole_.size());

    auto concat_and_add_to_ret = [&](std::vector<Chunk>&& chunks) {
        if (chunks.empty()) {
            return;
        } else if (chunks.size() == 1) {
            ret.emplace_back(std::move(chunks[0]));
        } else {
            ret.emplace_back(Chunk::concat(std::move(chunks), chunk_id_gen(), stream, br)
            );
        }
    };

    // Iterate through the outer map
    auto pid_it = pigeonhole_.begin();
    while (pid_it != pigeonhole_.end()) {
        // Iterate through the inner map
        auto& chunks = pid_it->second;

        std::vector<Chunk> ready_chunks;
        ready_chunks.reserve(chunks.size());
        size_t concat_size = 0;

        for (auto chunk_it = chunks.begin(); chunk_it != chunks.end();) {
            auto& chunk = chunk_it->second;
            if (chunk.is_ready()) {
                if (chunk.n_messages() >= 1) {
                    // chunk has been already concatenated
                    ret.emplace_back(std::move(chunk));
                    chunk_it = chunks.erase(chunk_it);
                    continue;
                }
                // if adding current chunk exceeds the max concat size,
                // concatenate the current chunks and add them to the ret
                if (concat_size + chunk.concat_data_size() >= max_concat_size) {
                    concat_and_add_to_ret(std::move(ready_chunks));
                    concat_size = 0;
                } else {
                    // add current chunk to the ready chunks
                    concat_size += chunk.concat_data_size();
                    ready_chunks.emplace_back(std::move(chunk));
                }

                chunk_it = chunks.erase(chunk_it);
            } else {
                ++chunk_it;
            }
        }

        concat_and_add_to_ret(std::move(ready_chunks));

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
            if (!chunk.is_control_message(0) && chunk.data_memory_type() == mem_type) {
                ret.emplace_back(key, cid, chunk.concat_data_size());
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
            if (chunk.is_control_message(0)) {
                ss << "EOP" << chunk.expected_num_chunks(0) << ", ";
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
