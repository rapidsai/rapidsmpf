/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::shuffler::detail {

namespace {

/// @brief Whether a chunk has a non-empty data buffer (e.g. not a control message).
bool has_data(Chunk const& chunk) {
    return chunk.data_size() > 0 && chunk.is_data_buffer_set();
}

/// @brief Index of a memory type into a `MEMORY_TYPES`-sized array.
constexpr std::size_t mem_type_index(MemoryType mem_type) {
    return static_cast<std::size_t>(mem_type);
}

}  // namespace

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
    std::lock_guard const lock(mutex_);
    std::stringstream ss;
    ss << "ChunksToSend(";
    for (auto const& chunk : chunks_) {
        ss << *chunk << ", ";
    }
    ss << ")";
    return ss.str();
}

void ReceivedChunks::insert(Chunk&& chunk) {
    auto key = chunk.part_id();
    std::lock_guard const lock(mutex_);
    if (has_data(chunk)) {
        data_sizes_[mem_type_index(chunk.data_memory_type())] += chunk.data_size();
    }
    pigeonhole_[key].emplace_back(std::move(chunk));
}

bool ReceivedChunks::is_empty(PartID pid) const {
    std::lock_guard const lock(mutex_);
    return !pigeonhole_.contains(pid);
}

std::vector<Chunk> ReceivedChunks::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    auto chunks = extract_value(pigeonhole_, pid);
    for (auto const& chunk : chunks) {
        if (has_data(chunk)) {
            data_sizes_[mem_type_index(chunk.data_memory_type())] -= chunk.data_size();
        }
    }
    return chunks;
}

bool ReceivedChunks::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

std::size_t ReceivedChunks::data_size(MemoryType mem_type) const {
    std::lock_guard const lock(mutex_);
    return data_sizes_[mem_type_index(mem_type)];
}

std::size_t ReceivedChunks::spill(BufferResource* br, std::size_t amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE(amount);
    std::lock_guard lock(mutex_);
    // Return early if there is no device data to spill.
    if (data_sizes_[mem_type_index(MemoryType::DEVICE)] == 0) {
        return 0;
    }
    // TODO: use a clever strategy to decided which chunks to spill.
    std::size_t total_spilled{0};
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& chunk : chunks) {
            if (!has_data(chunk) || chunk.data_memory_type() != MemoryType::DEVICE) {
                continue;
            }
            auto const size = chunk.data_size();
            auto reservation = br->reserve_or_fail(size, SPILL_TARGET_MEMORY_TYPES);
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), reservation));
            data_sizes_[mem_type_index(MemoryType::DEVICE)] -= size;
            data_sizes_[mem_type_index(chunk.data_memory_type())] += size;
            if ((total_spilled += size) >= amount) {
                break;
            }
        }
        if (total_spilled >= amount) {
            break;
        }
    }
    RAPIDSMPF_NVTX_MARKER("ReceivedChunks::spill::total_spilled", total_spilled);
    return total_spilled;
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
    ss << "data_sizes={";
    for (auto mem_type : MEMORY_TYPES) {
        ss << mem_type << "=" << format_nbytes(data_sizes_[mem_type_index(mem_type)])
           << ", ";
    }
    ss << "\b\b})";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
