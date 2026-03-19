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

void ChunksToSend::insert(std::unique_ptr<Chunk> c) {
    std::lock_guard lock(mutex_);
    chunks_.push_back(std::move(c));
}

std::vector<Chunk> ChunksToSend::extract_ready() {
    std::lock_guard lock(mutex_);
    std::vector<Chunk> result;
    for (auto&& chunk : chunks_) {
        if (!chunk->is_ready()) {
            break;
        }
        auto c = std::move(chunk);
        result.emplace_back(std::move(*c));
    }
    std::erase(chunks_, nullptr);
    return result;
}

bool ChunksToSend::empty() const {
    std::lock_guard lock(mutex_);
    return chunks_.empty();
}

std::string ChunksToSend::str() const {
    if (empty()) {
        return "ChunksToSend()";
    }
    std::lock_guard const lock(mutex_);
    std::stringstream ss;
    ss << "ChunksToSend(";
    for (auto const& chunk : chunks_) {
        ss << chunk << ", ";
    }
    ss << ")";
    return ss.str();
}

void PostBox::insert(Chunk&& chunk) {
    auto key = chunk.part_id();
    std::lock_guard const lock(mutex_);
    RAPIDSMPF_EXPECTS(
        pigeonhole_[key].emplace(chunk.chunk_id(), std::move(chunk)).second,
        "PostBox.insert(): chunk already exist"
    );
}

bool PostBox::is_empty(PartID pid) const {
    std::lock_guard const lock(mutex_);
    return !pigeonhole_.contains(pid);
}

Chunk PostBox::extract(PartID pid, ChunkID cid) {
    std::lock_guard const lock(mutex_);
    return extract_item(pigeonhole_[pid], cid).second;
}

std::unordered_map<ChunkID, Chunk> PostBox::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, pid);
}

std::unordered_map<ChunkID, Chunk> PostBox::extract_by_key(PartID key) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, key);
}

std::vector<Chunk> PostBox::extract_all_ready() {
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

bool PostBox::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

std::vector<std::tuple<PartID, ChunkID, std::size_t>> PostBox::search(
    MemoryType mem_type
) const {
    std::lock_guard const lock(mutex_);
    std::vector<std::tuple<PartID, ChunkID, std::size_t>> ret;
    for (auto& [key, chunks] : pigeonhole_) {
        for (auto& [cid, chunk] : chunks) {
            if (!chunk.is_control_message() && chunk.data_memory_type() == mem_type) {
                ret.emplace_back(key, cid, chunk.data_size());
            }
        }
    }
    return ret;
}

std::string PostBox::str() const {
    if (empty()) {
        return "PostBox()";
    }
    std::lock_guard const lock(mutex_);
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

}  // namespace rapidsmpf::shuffler::detail
