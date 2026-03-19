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

void ReceivedChunks::insert(Chunk&& chunk) {
    auto key = chunk.part_id();
    std::lock_guard const lock(mutex_);
    pigeonhole_[key].emplace_back(std::move(chunk));
}

bool ReceivedChunks::is_empty(PartID pid) const {
    std::lock_guard const lock(mutex_);
    return !pigeonhole_.contains(pid);
}

std::vector<Chunk> ReceivedChunks::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, pid);
}

bool ReceivedChunks::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

std::vector<std::reference_wrapper<Chunk>> ReceivedChunks::search(MemoryType mem_type) {
    std::lock_guard const lock(mutex_);
    std::vector<std::reference_wrapper<Chunk>> ret;
    for (auto& [key, chunks] : pigeonhole_) {
        for (auto& chunk : chunks) {
            if (!chunk.is_control_message() && chunk.data_memory_type() == mem_type) {
                ret.emplace_back(chunk);
            }
        }
    }
    return ret;
}

std::string ReceivedChunks::str() const {
    if (empty()) {
        return "ReceivedChunks()";
    }
    std::lock_guard const lock(mutex_);
    std::stringstream ss;
    ss << "ReceivedChunks(";
    for (auto const& [key, chunks] : pigeonhole_) {
        ss << "k=" << key << ": [";
        for (auto const& chunk : chunks) {
            ss << chunk << ", ";
        }
        ss << "\b\b], ";
    }
    ss << "\b\b)";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
