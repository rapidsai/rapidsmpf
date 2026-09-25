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

std::vector<Chunk> ChunksToSend::extract_and_restore(
    BufferResource* br, std::span<MemoryType const> memory_types
) {
    std::lock_guard lock(mutex_);
    std::vector<Chunk> result;
    for (auto&& chunk : chunks_) {
        if (!chunk->is_ready()) {
            break;
        }
        auto const restore = chunk->is_on_disk();
        if (restore) {
            auto reservation = br->try_reserve_or_spill(chunk->data_size(), memory_types);
            if (!reservation.has_value()) {
                // No addressable memory for the restore right now (e.g. the device is
                // at its limit with nothing left to spill because other users of the
                // buffer resource hold it). Leave this and all later chunks queued
                // and send what is ready; the progress loop retries on its next
                // iteration once memory has been released. Aborting here (the
                // previous behaviour) killed the process under transient pressure.
                break;
            }
            chunk->set_data_buffer(br->move(chunk->release_data_buffer(), *reservation));
            // The restore is an asynchronous disk->memory copy: the new buffer is not
            // ready (is_latest_write_done() == false) yet, and UCXX::send() requires a
            // ready buffer. Keep the chunk queued; the next progress iteration's
            // is_ready() check returns it once the copy has completed. Restoring is
            // also slow and adds addressable-memory pressure, so restore at most one
            // chunk per call.
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
    if (has_device_data(chunk)) {
        ++num_device_chunks_;
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
        if (has_device_data(chunk)) {
            --num_device_chunks_;
        }
    }
    return chunks;
}

bool ReceivedChunks::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

std::size_t ReceivedChunks::spill(
    BufferResource* br,
    std::size_t amount,
    std::span<MemoryType const> spillable_memory_types
) {
    if (amount == 0) {
        return 0;
    }

    RAPIDSMPF_NVTX_FUNC_RANGE(amount);
    std::lock_guard lock(mutex_);
    if (num_device_chunks_ == 0) {
        return 0;
    }
    // TODO: use a clever strategy to decided which chunks to spill.
    std::size_t total_spilled{0};
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& chunk : chunks) {
            if (!has_device_data(chunk)) {
                continue;
            }
            auto const size = chunk.data_size();
            auto reservation = br->try_reserve(size, spillable_memory_types);
            if (!reservation.has_value()) {
                continue;
            }
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), *reservation));
            --num_device_chunks_;
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
    ss << "\b\b)";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
