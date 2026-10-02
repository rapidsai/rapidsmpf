/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::shuffler::detail {

SendOrderPolicy parse_send_order_policy(std::string_view name) {
    auto const lower = to_lower(trim(name));
    if (lower == "global") {
        return SendOrderPolicy::Global;
    }
    if (lower == "rank") {
        return SendOrderPolicy::PerRank;
    }
    if (lower == "pid") {
        return SendOrderPolicy::PerPartition;
    }
    if (lower == "none") {
        return SendOrderPolicy::None;
    }
    RAPIDSMPF_FAIL(
        "invalid send order policy: \"" + std::string{name}
            + "\" (expected global, rank, pid or none)",
        std::invalid_argument
    );
}

SendOrderPolicy send_order_policy_from_env() {
    char const* env = std::getenv("RAPIDSMPF_SHUFFLER_SEND_ORDER_POLICY");
    if (env == nullptr || *env == '\0') {
        return SendOrderPolicy::Global;
    }
    return parse_send_order_policy(env);
}

std::ostream& operator<<(std::ostream& os, SendOrderPolicy policy) {
    switch (policy) {
    case SendOrderPolicy::Global:
        return os << "global";
    case SendOrderPolicy::PerRank:
        return os << "rank";
    case SendOrderPolicy::PerPartition:
        return os << "pid";
    case SendOrderPolicy::None:
        return os << "none";
    }
    return os << "unknown";
}

void ChunksToSend::insert(Rank dst, std::unique_ptr<Chunk> c) {
    std::lock_guard lock(mutex_);
    chunks_.emplace_back(dst, std::move(c));
}

std::vector<Chunk> ChunksToSend::extract_ready(ExtractStats* stats) {
    return extract(nullptr, {}, stats);
}

std::vector<Chunk> ChunksToSend::extract_and_restore(
    BufferResource* br, std::span<MemoryType const> memory_types, ExtractStats* stats
) {
    RAPIDSMPF_EXPECTS(br != nullptr, "the buffer resource pointer cannot be NULL");
    return extract(br, memory_types, stats);
}

std::vector<Chunk> ChunksToSend::extract(
    BufferResource* br, std::span<MemoryType const> memory_types, ExtractStats* stats
) {
    std::lock_guard lock(mutex_);
    ExtractStats local_stats;
    std::vector<Chunk> result;

    // Keys (destination rank or partition ID) of chunks that could not be extracted. A
    // later chunk with a blocked key is held back so that, for each key, chunks leave in
    // insertion order.
    std::unordered_set<std::uint64_t> blocked_keys;
    auto const key_of = [&](Rank dst, Chunk const& chunk) -> std::uint64_t {
        return policy_ == SendOrderPolicy::PerRank
                   ? safe_cast<std::uint64_t>(dst)
                   : static_cast<std::uint64_t>(chunk.part_id());
    };
    // Restoring is slow and adds pressure on addressable memory, so restore at most one
    // disk-backed chunk per call.
    std::size_t restores_left = 1;

    for (auto&& [dst, chunk] : chunks_) {
        auto const key = key_of(dst, *chunk);
        if (policy_ == SendOrderPolicy::Global && !blocked_keys.empty()) {
            break;
        }
        if (policy_ != SendOrderPolicy::None && blocked_keys.contains(key)) {
            continue;
        }
        if (!chunk->is_ready()) {
            ++local_stats.not_ready;
            blocked_keys.insert(key);
            continue;
        }
        if (br != nullptr && chunk->is_on_disk()) {
            std::optional<MemoryReservation> reservation;
            if (restores_left > 0) {
                reservation = br->try_reserve_or_spill(chunk->data_size(), memory_types);
            }
            if (!reservation.has_value()) {
                RAPIDSMPF_EXPECTS(
                    policy_ != SendOrderPolicy::Global,
                    "failed to reserve addressable memory for an outgoing disk-backed "
                    "chunk",
                    std::runtime_error
                );
                ++local_stats.restore_deferred;
                blocked_keys.insert(key);
                continue;
            }
            chunk->set_data_buffer(br->move(chunk->release_data_buffer(), *reservation));
            --restores_left;
            ++local_stats.restored;
            if (policy_ == SendOrderPolicy::Global) {
                // Preserve the original behavior: stop after the first restore.
                result.emplace_back(std::move(*chunk));
                chunk.reset();
                break;
            }
        }
        result.emplace_back(std::move(*chunk));
        chunk.reset();
    }
    std::erase_if(chunks_, [](auto const& entry) { return entry.second == nullptr; });
    if (stats != nullptr) {
        // Every chunk left behind that was not itself blocking is held back by one.
        local_stats.blocked =
            chunks_.size() - local_stats.not_ready - local_stats.restore_deferred;
        *stats = local_stats;
    }
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
    for (auto const& [dst, chunk] : chunks_) {
        ss << "dst=" << dst << ": " << *chunk << ", ";
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
    // TODO: use a clever strategy to decided which chunks to spill.
    std::size_t total_spilled{0};
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& chunk : chunks) {
            if (chunk.data_size() == 0 || !chunk.is_data_buffer_set()
                || chunk.data_memory_type() != MemoryType::DEVICE)
            {
                continue;
            }
            auto const size = chunk.data_size();
            auto reservation = br->try_reserve(size, spillable_memory_types);
            if (!reservation.has_value()) {
                continue;
            }
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), *reservation));
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
