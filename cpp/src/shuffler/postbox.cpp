/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {


void PostBox::insert(Chunk&& chunk) {
    std::lock_guard const lock(mutex_);
    auto [_, inserted] = pigeonhole_[chunk.pid].insert({chunk.cid, std::move(chunk)});
    RAPIDSMPF_EXPECTS(inserted, "PostBox.insert(): chunk already exist");
}

Chunk PostBox::extract(PartID pid, ChunkID cid) {
    std::lock_guard const lock(mutex_);
    return extract_item(pigeonhole_.at(pid), cid).second;
}

std::unordered_map<ChunkID, Chunk> PostBox::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, pid);
}

std::vector<Chunk> PostBox::extract_all() {
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

bool PostBox::empty() const {
    return pigeonhole_.empty();
}

std::vector<std::tuple<PartID, ChunkID, std::size_t>> PostBox::search(MemoryType mem_type
) const {
    std::lock_guard const lock(mutex_);
    std::vector<std::tuple<PartID, ChunkID, std::size_t>> ret;
    for (auto& [pid, chunks] : pigeonhole_) {
        for (auto& [cid, chunk] : chunks) {
            if (chunk.gpu_data && chunk.gpu_data->mem_type() == mem_type) {
                ret.emplace_back(pid, cid, chunk.gpu_data->size);
            }
        }
    }
    return ret;
}

std::string PostBox::str() const {
    if (empty()) {
        return "PostBox()";
    }
    std::stringstream ss;
    ss << "PostBox(";
    for (auto const& [pid, chunks] : pigeonhole_) {
        ss << "p" << pid << ": [";
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

PostBoxByRank::PostBoxByRank(size_t num_ranks) {
    // Pre-allocate space for each rank's vector
    pigeonhole_.reserve(num_ranks);
}

void PostBoxByRank::insert(Rank rank, Chunk&& chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    pigeonhole_[rank].emplace_back(std::move(chunk));
}

ChunkVector PostBoxByRank::extract(Rank rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pigeonhole_.find(rank);
    if (it == pigeonhole_.end()) {
        return ChunkVector{};
    }
    ChunkVector chunks = std::move(it->second);
    pigeonhole_.erase(it);
    return chunks;
}

std::vector<ChunkVector> PostBoxByRank::extract_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ChunkVector> all_chunks;
    all_chunks.reserve(pigeonhole_.size());

    for (auto& [rank, chunks] : pigeonhole_) {
        all_chunks.emplace_back(std::move(chunks));
    }
    pigeonhole_.clear();
    return all_chunks;
}

bool PostBoxByRank::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pigeonhole_.empty();
}

}  // namespace rapidsmpf::shuffler::detail
