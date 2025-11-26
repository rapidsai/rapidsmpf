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

    auto& map_value = pigeonhole_.at(key);
    std::lock_guard lock(map_value.mutex);
    // if (map_value.chunks.empty()) {
    //     // this key is currently empty. So increment the non-empty key count.
    //     RAPIDSMPF_EXPECTS(
    //         n_non_empty_keys_.fetch_add(1, std::memory_order_relaxed) + 1
    //             <= pigeonhole_.size(),
    //         "PostBox.insert(): n_non_empty_keys_ is already at the maximum"
    //     );
    // }
    n_chunks.fetch_add(1, std::memory_order_relaxed);
    if (chunk.is_data_buffer_set() && chunk.data_memory_type() == MemoryType::HOST) {
        map_value.ready_chunks.emplace_back(std::move(chunk));
    } else {
        map_value.ready_chunks.emplace_front(std::move(chunk));
    }
}

template <typename KeyType>
bool PostBox<KeyType>::is_empty(PartID pid) const {
    auto& map_value = pigeonhole_.at(key_map_fn_(pid));
    std::lock_guard lock(map_value.mutex);
    // return map_value.chunks.empty();
    return map_value.is_empty_unsafe();
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract(PartID pid) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    return extract_by_key(key_map_fn_(pid));
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract_by_key(KeyType key) {
    auto& map_value = pigeonhole_.at(key);
    std::lock_guard lock(map_value.mutex);
    RAPIDSMPF_EXPECTS(
        !map_value.is_empty_unsafe(), "PostBox.extract(): partition is empty"
    );

    std::vector<Chunk> ret(
        std::make_move_iterator(map_value.ready_chunks.begin()),
        std::make_move_iterator(map_value.ready_chunks.end())
    );
    map_value.ready_chunks.clear();

    // RAPIDSMPF_EXPECTS(
    //     n_non_empty_keys_.fetch_sub(1, std::memory_order_relaxed) > 0,
    //     "PostBox.extract(): n_non_empty_keys_ is already 0"
    // );
    RAPIDSMPF_EXPECTS(
        n_chunks.fetch_sub(ret.size(), std::memory_order_relaxed) >= ret.size(),
        "PostBox.extract(): n_chunks is negative"
    );
    return ret;
}

template <typename KeyType>
std::vector<Chunk> PostBox<KeyType>::extract_all_ready() {
    std::vector<Chunk> ret;

    // Iterate through the outer map
    for (auto& [key, map_value] : pigeonhole_) {
        std::unique_lock lock(map_value.mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            continue;
        }

        // // Partition: non-ready chunks first, ready chunks at the end
        // auto partition_point =
        //     std::ranges::partition(map_value.chunks, [](const Chunk& c) {
        //         return !c.is_ready();
        //     }).begin();

        // // if the chunks are available and all are ready, then all chunks will be
        // // extracted
        // if (map_value.chunks.begin() == partition_point
        //     && partition_point != map_value.chunks.end())
        // {
        //     RAPIDSMPF_EXPECTS(
        //         n_non_empty_keys_.fetch_sub(1, std::memory_order_relaxed) > 0,
        //         "PostBox.extract_all_ready(): n_non_empty_keys_ is already 0"
        //     );
        // }

        // // Move ready chunks to result
        // ret.insert(
        //     ret.end(),
        //     std::make_move_iterator(partition_point),
        //     std::make_move_iterator(map_value.chunks.end())
        // );

        // // Remove ready chunks from the vector
        // map_value.chunks.erase(partition_point, map_value.chunks.end());

        for (auto it = map_value.ready_chunks.begin();
             it != map_value.ready_chunks.end();)
        {
            if (it->is_ready()) {
                ret.emplace_back(std::move(*it));
                it = map_value.ready_chunks.erase(it);
                RAPIDSMPF_EXPECTS(
                    n_chunks.fetch_sub(1, std::memory_order_relaxed) >= 1,
                    "PostBox.extract_all_ready(): n_chunks is negative"
                );
            } else {
                ++it;
            }
        }
    }
    return ret;
}

template <typename KeyType>
bool PostBox<KeyType>::empty() const {
    return n_chunks.load(std::memory_order_acquire) == 0;
}

template <typename KeyType>
size_t PostBox<KeyType>::spill(
    BufferResource* br, Communicator::Logger& /* log */, size_t amount
) {
    RAPIDSMPF_NVTX_SCOPED_RANGE("spill-inside-postbox");

    // individually lock each key and spill the chunks in it. If we are unable to lock the
    // key, then it will be skipped.
    size_t total_spilled = 0;

    // auto it_start = pigeonhole_.begin();
    // std::advance(it_start, dist_(rng_));


    // auto spill_chunks = [&](auto& map_value) {
    //     std::unique_lock lock(map_value.mutex, std::try_to_lock);

    //     if (!lock) {
    //         return false;
    //     }

    //     for (auto& chunk : map_value.chunks) {
    //         if (chunk.is_data_buffer_set()
    //             && chunk.data_memory_type() == MemoryType::DEVICE)
    //         {
    //             size_t size = chunk.concat_data_size();
    //             auto [host_reservation, host_overbooking] =
    //                 br->reserve(MemoryType::HOST, size, true);
    //             if (host_overbooking > 0) {
    //                 log.warn(
    //                     "Cannot spill to host because of host memory overbooking: ",
    //                     format_nbytes(host_overbooking)
    //                 );
    //                 continue;
    //             }
    //             chunk.set_data_buffer(
    //                 br->move(chunk.release_data_buffer(), host_reservation)
    //             );
    //             total_spilled += size;
    //             if (total_spilled >= amount) {
    //                 return true;
    //             }
    //         }
    //     }
    //     return false;
    // };

    // for (auto it = it_start; it != pigeonhole_.end(); ++it) {
    //     auto& [key, map_value] = *it;
    //     // std::unique_lock lock(map_value.mutex, std::try_to_lock);
    //     // if (!lock) {  // skip to the next key
    //     //     continue;
    //     // }

    //     // for (auto& chunk : map_value.chunks) {
    //     //     if (chunk.is_data_buffer_set()
    //     //         && chunk.data_memory_type() == MemoryType::DEVICE)
    //     //     {
    //     //         size_t size = chunk.concat_data_size();
    //     //         auto [host_reservation, host_overbooking] =
    //     //             br->reserve(MemoryType::HOST, size, true);
    //     //         if (host_overbooking > 0) {
    //     //             log.warn(
    //     //                 "Cannot spill to host because of host memory overbooking: ",
    //     //                 format_nbytes(host_overbooking)
    //     //             );
    //     //             continue;
    //     //         }
    //     //         chunk.set_data_buffer(
    //     //             br->move(chunk.release_data_buffer(), host_reservation)
    //     //         );
    //     //         total_spilled += size;
    //     //         if (total_spilled >= amount) {
    //     //             break;
    //     //         }
    //     //     }
    //     // }
    //     if (spill_chunks(map_value)) {
    //         break;
    //     }
    // }

    // for (auto it = pigeonhole_.begin(); it != it_start; ++it) {
    //     auto& [key, map_value] = *it;
    //     if (spill_chunks(map_value)) {
    //         break;
    //     }
    // }

    for (auto& [key, map_value] : pigeonhole_) {
        std::unique_lock lock(map_value.mutex, std::try_to_lock);
        if (!lock) {  // skip to the next key
            continue;
        }

        std::vector<Chunk> spillable_chunks;
        for (auto it = map_value.ready_chunks.begin();
             it != map_value.ready_chunks.end();)
        {
            auto& chunk = *it;
            if (chunk.is_data_buffer_set()
                && chunk.data_memory_type() == MemoryType::DEVICE)
            {
                size_t size = chunk.concat_data_size();
                spillable_chunks.emplace_back(std::move(chunk));
                it = map_value.ready_chunks.erase(it);
                total_spilled += size;
                if (total_spilled >= amount) {
                    break;
                }
            } else {
                ++it;
            }
        }
        map_value.n_spilling_chunks += spillable_chunks.size();
        // release lock
        lock.unlock();

        // spill the chunks to host memory
        while (!spillable_chunks.empty()) {
            auto chunk = std::move(spillable_chunks.back());
            spillable_chunks.pop_back();
            size_t size = chunk.concat_data_size();
            auto [host_reservation, host_overbooking] =
                br->reserve(MemoryType::HOST, size, true);
            RAPIDSMPF_EXPECTS(
                host_overbooking == 0,
                "Cannot spill to host because of host memory overbooking: "
                    + std::to_string(host_overbooking)
            );
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), host_reservation)
            );

            lock.lock();
            map_value.ready_chunks.emplace_back(std::move(chunk));
            map_value.n_spilling_chunks--;
            lock.unlock();
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
        ss << "k=" << key << " nspill=" << map_value.n_spilling_chunks << ":[";
        for (auto const& chunk : map_value.ready_chunks) {
            // assert(cid == chunk.chunk_id());
            if (chunk.is_control_message(0)) {
                ss << "EOP" << chunk.expected_num_chunks(0) << ", ";
            } else {
                ss << chunk.chunk_id() << ", ";
            }
        }
        ss << (map_value.ready_chunks.empty() ? "], " : "\b\b], ");
    }
    ss << "\b\b)";
    return ss.str();
}

// Explicit instantiation for PartID and Rank
template class PostBox<PartID>;
template class PostBox<Rank>;

}  // namespace rapidsmpf::shuffler::detail
