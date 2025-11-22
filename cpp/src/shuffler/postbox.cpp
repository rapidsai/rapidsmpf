/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <ranges>
#include <sstream>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/nvtx.hpp>
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
    map_value.chunks.push_back(std::move(chunk));
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
    std::lock_guard lock(map_value.mutex);
    RAPIDSMPF_EXPECTS(!map_value.chunks.empty(), "PostBox.extract(): partition is empty");

    std::vector<Chunk> ret = std::move(map_value.chunks);
    map_value.chunks.clear();

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

        // Partition: non-ready chunks first, ready chunks at the end
        auto partition_point =
            std::ranges::partition(map_value.chunks, [](const Chunk& c) {
                return !c.is_ready();
            }).begin();

        // if the chunks are available and all are ready, then all chunks will be
        // extracted
        if (map_value.chunks.begin() == partition_point
            && partition_point != map_value.chunks.end())
        {
            RAPIDSMPF_EXPECTS(
                n_non_empty_keys_.fetch_sub(1, std::memory_order_relaxed) > 0,
                "PostBox.extract_all_ready(): n_non_empty_keys_ is already 0"
            );
        }

        // Move ready chunks to result
        ret.insert(
            ret.end(),
            std::make_move_iterator(partition_point),
            std::make_move_iterator(map_value.chunks.end())
        );

        // Remove ready chunks from the vector
        map_value.chunks.erase(partition_point, map_value.chunks.end());
    }
    return ret;
}

template <typename KeyType>
bool PostBox<KeyType>::empty() const {
    return n_non_empty_keys_.load(std::memory_order_acquire) == 0;
}

template <typename KeyType>
size_t PostBox<KeyType>::spill(
    BufferResource* br, Communicator::Logger& log, size_t amount
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();

    // individually lock each key and spill the chunks in it. If we are unable to lock the
    // key, then it will be skipped.
    size_t total_spilled = 0;
    for (auto& [key, map_value] : pigeonhole_) {
        std::unique_lock lock(map_value.mutex, std::try_to_lock);
        if (lock) {  // now all chunks in this key are locked
            for (auto& chunk : map_value.chunks) {
                if (chunk.is_data_buffer_set()
                    && chunk.data_memory_type() == MemoryType::DEVICE)
                {
                    size_t size = chunk.concat_data_size();
                    auto [host_reservation, host_overbooking] =
                        br->reserve(MemoryType::HOST, size, true);
                    if (host_overbooking > 0) {
                        log.warn(
                            "Cannot spill to host because of host memory overbooking: ",
                            format_nbytes(host_overbooking)
                        );
                        continue;
                    }
                    chunk.set_data_buffer(
                        br->move(chunk.release_data_buffer(), host_reservation)
                    );
                    total_spilled += size;
                    if (total_spilled >= amount) {
                        break;
                    }
                }
            }
        }
    }

    return total_spilled;
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
