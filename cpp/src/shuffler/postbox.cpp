/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {


PostBox::PostBox(Rank nranks, std::function<Rank(PartID)>&& partition_owner)
    : partition_owner_{std::move(partition_owner)} {
    pigeonhole_.reserve(static_cast<std::size_t>(nranks));
}

void PostBox::insert(Chunk&& chunk) {
    std::lock_guard const lock(mutex_);
    Rank owner = partition_owner_(chunk.pid);
    auto [_, inserted] = chunk_ids_.emplace(chunk.cid);
    RAPIDSMPF_EXPECTS(
        inserted, "PostBox.insert(): chunk already exist " + std::to_string(chunk.cid)
    );

    // insert into the rank-based pigeonhole
    pigeonhole_[owner].emplace_back(std::move(chunk));
}

Chunk PostBox::extract(PartID pid, ChunkID cid) {
    std::lock_guard const lock(mutex_);
    std::cout << "extracting chunk " << cid << " pid " << pid << std::endl;
    // check if chunk is in the chunk_ids_
    auto it = chunk_ids_.find(cid);
    RAPIDSMPF_EXPECTS(
        it != chunk_ids_.end(),
        "PostBox.extract(): chunk not found " + std::to_string(cid)
    );

    // chunk is available, therefore no check the availability in the pigeonhole
    auto& chunks = pigeonhole_[partition_owner_(pid)];
    // iterate over the chunks in the rank to find the chunk with the given cid
    auto cid_it = std::find_if(chunks.begin(), chunks.end(), [cid](const auto& chunk) {
        return chunk.cid == cid;
    });
    auto chunk = std::move(*cid_it);
    chunks.erase(cid_it);  // remove the chunk from the list


    chunk_ids_.erase(cid);  // remove the chunk from the chunk_ids_

    return chunk;
}

std::unordered_map<ChunkID, Chunk> PostBox::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    auto& chunks = pigeonhole_[partition_owner_(pid)];

    std::unordered_map<ChunkID, Chunk> ret;

    // iterate over the chunks and extract chunks with pid
    for (auto it = chunks.begin(); it != chunks.end();) {
        auto chunk_id = it->cid;
        if (it->pid == pid) {
            ret.emplace(chunk_id, std::move(*it));
            chunk_ids_.erase(chunk_id);
            it = chunks.erase(it);
        } else {
            ++it;
        }
    }

    return ret;
}

std::vector<Chunk> PostBox::extract_all() {
    std::lock_guard const lock(mutex_);
    std::vector<Chunk> ret;
    ret.reserve(chunk_ids_.size());

    // iterate over the pigeonhole and extract all chunks
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& chunk : chunks) {
            ret.push_back(std::move(chunk));
        }
    }
    pigeonhole_.clear();
    chunk_ids_.clear();
    return ret;
}

std::list<Chunk> PostBox::extract_for_rank(std::optional<Rank> rank) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = rank.has_value() ? pigeonhole_.find(*rank) : pigeonhole_.begin();
    if (it == pigeonhole_.end()) {  // rank not found or empty
        return {};
    }
    std::list<Chunk> chunks = std::move(it->second);
    for (auto& chunk : chunks) {
        chunk_ids_.erase(chunk.cid);
    }
    pigeonhole_.erase(it);
    return chunks;
}

bool PostBox::empty() const {
    return pigeonhole_.empty();
}

std::vector<std::tuple<PartID, ChunkID, std::size_t>> PostBox::search(MemoryType mem_type
) const {
    std::lock_guard const lock(mutex_);
    std::vector<std::tuple<PartID, ChunkID, std::size_t>> ret;
    for (auto const& [pid, chunks] : pigeonhole_) {
        for (auto const& chunk : chunks) {
            if (chunk.gpu_data && chunk.gpu_data->mem_type() == mem_type) {
                ret.emplace_back(pid, chunk.cid, chunk.gpu_data->size);
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
        for (auto const& chunk : chunks) {
            if (chunk.expected_num_chunks) {
                ss << "EOP" << chunk.expected_num_chunks << ", ";
            } else {
                ss << chunk.cid << ", ";
            }
        }
        ss << "\b\b], ";
    }
    ss << "\b\b)";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
