/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda/stream>

#include <rapidsmpf/coll/utils.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::coll::detail {

Chunk::Chunk(
    ChunkID id,
    Rank destination,
    std::unique_ptr<std::vector<std::uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : id_{id},
      destination_{destination},
      metadata_{std::move(metadata)},
      data_{std::move(data)},
      data_size_{data_ ? data_->size : 0} {
    RAPIDSMPF_EXPECTS(
        (metadata_ == nullptr) == (data_ == nullptr),
        "One of metadata or data is nullptr, but both should be valid pointers",
        std::logic_error
    );
    RAPIDSMPF_EXPECTS(
        metadata_ && data_,
        "Non-finish chunk must have metadata and data",
        std::invalid_argument
    );
}

Chunk::Chunk(ChunkID id, Rank destination)
    : id_{id},
      destination_{destination},
      metadata_{nullptr},
      data_{nullptr},
      data_size_{0} {}

bool Chunk::is_ready() const noexcept {
    return data_size_ == 0 || (data_ && data_->is_latest_write_done());
}

MemoryType Chunk::memory_type() const noexcept {
    return data_ == nullptr ? MemoryType::HOST : data_->mem_type();
}

bool Chunk::is_finish() const noexcept {
    return data_ == nullptr && metadata_ == nullptr;
}

ChunkID Chunk::id() const noexcept {
    return id_;
}

ChunkID Chunk::sequence() const noexcept {
    return id() & ((static_cast<std::uint64_t>(1) << ID_BITS) - 1);
}

Rank Chunk::origin() const noexcept {
    return id() >> ID_BITS;
}

Rank Chunk::destination() const noexcept {
    return destination_;
}

std::uint64_t Chunk::data_size() const noexcept {
    return data_size_;
}

std::uint64_t Chunk::metadata_size() const noexcept {
    return metadata_ ? metadata_->size() : 0;
}

std::unique_ptr<Chunk> Chunk::from_packed_data(
    std::uint64_t sequence, Rank origin, Rank destination, PackedData&& packed_data
) {
    return std::unique_ptr<Chunk>(new Chunk(
        chunk_id(sequence, origin),
        destination,
        std::move(packed_data.metadata),
        std::move(packed_data.data)
    ));
}

std::unique_ptr<Chunk> Chunk::from_empty(
    std::uint64_t sequence, Rank origin, Rank destination
) {
    return std::unique_ptr<Chunk>(new Chunk(chunk_id(sequence, origin), destination));
}

constexpr ChunkID Chunk::chunk_id(std::uint64_t sequence, Rank origin) {
    return (static_cast<std::uint64_t>(origin) << ID_BITS)
           | static_cast<std::uint64_t>(sequence);
}

std::unique_ptr<std::vector<std::uint8_t>> Chunk::serialize() const {
    std::size_t size = sizeof(ChunkID);
    if (!is_finish()) {
        size += sizeof(data_size_) + metadata_size();
    }
    auto result = std::make_unique<std::vector<std::uint8_t>>(size);
    std::memcpy(result->data(), &id_, sizeof(ChunkID));
    if (!is_finish()) {
        std::memcpy(result->data() + sizeof(ChunkID), &data_size_, sizeof(data_size_));
        if (metadata_size() > 0) {
            std::memcpy(
                result->data() + sizeof(ChunkID) + sizeof(data_size_),
                metadata_->data(),
                metadata_->size()
            );
        }
    }
    return result;
}

std::unique_ptr<Chunk> Chunk::deserialize(
    std::vector<std::uint8_t>& data,
    BufferResource* br,
    std::span<MemoryType const> memory_types
) {
    ChunkID id;
    std::uint64_t data_size;
    std::memcpy(&id, data.data(), sizeof(ChunkID));
    if (data.size() == sizeof(id)) {
        return std::unique_ptr<Chunk>(new Chunk(id, Chunk::INVALID_RANK));
    }
    std::memcpy(&data_size, data.data() + sizeof(ChunkID), sizeof(data_size));
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        data.size() - sizeof(ChunkID) - sizeof(data_size)
    );
    std::memcpy(
        metadata->data(),
        data.data() + sizeof(ChunkID) + sizeof(data_size),
        metadata->size()
    );
    auto reservation = br->try_reserve_or_spill(data_size, memory_types);
    RAPIDSMPF_EXPECTS(
        reservation.has_value(),
        "failed to reserve addressable memory for an incoming allgather chunk",
        std::runtime_error
    );
    return std::unique_ptr<Chunk>(new Chunk(
        id,
        Chunk::INVALID_RANK,
        std::move(metadata),
        br->make_buffer(br->stream_pool()->get_stream(), std::move(*reservation))
    ));
}

PackedData Chunk::release() {
    RAPIDSMPF_EXPECTS(metadata_ && data_, "Can't release Chunk with no metadata or data");
    return {std::move(metadata_), std::move(data_)};
}

std::unique_ptr<Buffer> Chunk::release_data_buffer() noexcept {
    return std::move(data_);
}

void Chunk::attach_data_buffer(std::unique_ptr<Buffer> data) {
    RAPIDSMPF_EXPECTS(data->size == data_size_, "Mismatching data size");
    RAPIDSMPF_EXPECTS(data_ == nullptr, "Chunk already has data");
    data_ = std::move(data);
}

void PostBox::insert(std::unique_ptr<Chunk> chunk) {
    std::lock_guard lock(mutex_);
    chunks_.emplace_back(std::move(chunk));
}

void PostBox::insert(std::vector<std::unique_ptr<Chunk>>&& chunks) {
    std::lock_guard lock(mutex_);
    std::ranges::for_each(chunks, [&](auto&& chunk) {
        chunks_.emplace_back(std::move(chunk));
    });
}

std::vector<std::unique_ptr<Chunk>> PostBox::extract_ready() {
    std::lock_guard lock(mutex_);
    std::vector<std::unique_ptr<Chunk>> result;
    for (auto&& chunk : chunks_) {
        if (!chunk->is_ready()) {
            continue;
        }
        result.emplace_back(std::move(chunk));
    }
    std::erase(chunks_, nullptr);
    return result;
}

std::vector<std::unique_ptr<Chunk>> PostBox::extract_and_restore(
    BufferResource* br, std::span<MemoryType const> memory_types
) {
    std::vector<std::unique_ptr<Chunk>> result;
    std::vector<std::unique_ptr<Chunk>> disk_chunks;
    std::size_t total_disk_size{0};
    {
        std::lock_guard lock(mutex_);
        for (auto&& chunk : chunks_) {
            if (!chunk->is_ready()) {
                continue;
            }
            if (chunk->memory_type() == MemoryType::DISK) {
                auto const size = safe_cast<std::size_t>(chunk->data_size());
                total_disk_size =
                    size > std::numeric_limits<std::size_t>::max() - total_disk_size
                        ? std::numeric_limits<std::size_t>::max()
                        : total_disk_size + size;
                disk_chunks.emplace_back(std::move(chunk));
            } else {
                result.emplace_back(std::move(chunk));
            }
        }
        std::erase(chunks_, nullptr);
    }

    if (disk_chunks.empty()) {
        return result;
    }

    std::ranges::sort(disk_chunks, std::ranges::less{}, [](auto const& chunk) {
        return chunk->data_size();
    });

    // The availability values are advisory. The subsequent reservation is the
    // authority because another thread may reserve memory after this snapshot.
    auto const available = br->memory_available_for_reservation();
    std::ptrdiff_t restore_count{0};
    auto const smallest_size = safe_cast<std::size_t>(disk_chunks.front()->data_size());
    auto const memory_type = std::ranges::find_if(memory_types, [&](auto const mem_type) {
        auto const value = available[static_cast<std::size_t>(mem_type)];
        return value >= 0 && safe_cast<std::size_t>(value) >= smallest_size;
    });

    // Reserve the full batch in the first tier that can make progress, then restore the
    // smallest disk chunks that fit after releasing any overbooking.
    if (memory_type != memory_types.end()) {
        auto const reservation_size = std::min(
            total_disk_size,
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
        );
        auto [reservation, overbooking] =
            br->reserve(*memory_type, reservation_size, AllowOverbooking::YES);

        // Keep only the portion of the full-batch reservation that is actually
        // available.
        br->release(reservation, std::min(overbooking, reservation.size()));

        for (auto& chunk : disk_chunks) {
            if (reservation.size() < safe_cast<std::size_t>(chunk->data_size())) {
                break;
            }
            chunk->attach_data_buffer(
                br->move(chunk->release_data_buffer(), reservation)
            );
            result.emplace_back(std::move(chunk));
            ++restore_count;
        }
    }

    // No tier could fit the smallest chunk, or the bulk reservation lost a race and
    // was trimmed below its size. Ask the spill manager to make enough space for one
    // chunk so progress is still guaranteed whenever any configured tier can satisfy it.
    if (restore_count == 0) {
        auto reservation =
            br->try_reserve_or_spill(disk_chunks.front()->data_size(), memory_types);
        if (!reservation.has_value()) {
            insert(std::move(disk_chunks));
            RAPIDSMPF_FAIL(
                "failed to reserve addressable memory for an outgoing disk-backed "
                "allgather chunk",
                std::runtime_error
            );
        }
        auto& chunk = disk_chunks.front();
        chunk->attach_data_buffer(br->move(chunk->release_data_buffer(), *reservation));
        result.emplace_back(std::move(chunk));
        restore_count = 1;
    }
    // erase restored chunk indices
    disk_chunks.erase(disk_chunks.begin(), std::next(disk_chunks.begin(), restore_count));
    if (!disk_chunks.empty()) {
        insert(std::move(disk_chunks));
    }
    return result;
}

std::vector<std::unique_ptr<Chunk>> PostBox::extract() {
    std::lock_guard lock(mutex_);
    return std::exchange(chunks_, {});
}

std::size_t PostBox::size() const noexcept {
    std::lock_guard lock(mutex_);
    return chunks_.size();
}

bool PostBox::empty() const noexcept {
    std::lock_guard lock(mutex_);
    return chunks_.empty();
}

std::size_t PostBox::spill(
    BufferResource* br,
    std::size_t amount,
    std::span<MemoryType const> spillable_memory_types
) {
    std::lock_guard lock(mutex_);
    if (amount == 0 || spillable_memory_types.empty()) {
        return 0;
    }
    std::vector<Chunk*> spillable_chunks;
    std::size_t max_spillable{0};
    std::size_t total_spilled{0};
    for (auto&& chunk : chunks_) {
        if (chunk->memory_type() == MemoryType::DEVICE) {
            spillable_chunks.push_back(chunk.get());
            max_spillable += chunk->data_size();
        }
    }
    auto spill_chunk = [&](Chunk* chunk) -> std::size_t {
        auto reservation = br->try_reserve(chunk->data_size(), spillable_memory_types);
        if (!reservation.has_value()) {
            return 0;
        }
        chunk->attach_data_buffer(br->move(chunk->release_data_buffer(), *reservation));
        return chunk->data_size();
    };
    if (max_spillable < amount) {
        // need to spill everything.
        for (auto&& chunk : spillable_chunks) {
            total_spilled += spill_chunk(chunk);
        }
        return total_spilled;
    }
    std::ranges::sort(spillable_chunks, std::less{}, [](Chunk* chunk) {
        return chunk->data_size();
    });
    // Try and spill the minimum number of buffers summing to the
    // amount we need while minimising the amount of data we need to
    // spill.
    while (!spillable_chunks.empty()) {
        auto pos = std::ranges::lower_bound(
            spillable_chunks, amount - total_spilled, std::less{}, [](Chunk* chunk) {
                return chunk->data_size();
            }
        );

        Chunk* chunk;
        if (pos == spillable_chunks.end()) {
            // No single chunk can satisfy remaining amount, so spill largest chunk.
            chunk = spillable_chunks.back();
            spillable_chunks.pop_back();
        } else {
            chunk = *pos;
            spillable_chunks.erase(pos);
        }
        total_spilled += spill_chunk(chunk);
        if (total_spilled >= amount) {
            break;
        }
    }
    return total_spilled;
}

std::vector<std::unique_ptr<Chunk>> test_some(
    std::vector<std::unique_ptr<Chunk>>& chunks,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    Communicator* comm
) {
    RAPIDSMPF_EXPECTS(
        chunks.size() == futures.size(), "Mismatching size for chunks and futures"
    );
    if (chunks.empty()) {
        return {};
    }
    auto [complete_futures, indices] = comm->test_some(futures);
    std::vector<std::unique_ptr<Chunk>> result;
    result.reserve(complete_futures.size());
    std::ranges::transform(
        indices, complete_futures, std::back_inserter(result), [&](auto i, auto&& fut) {
            auto chunk = std::move(chunks[i]);
            chunk->attach_data_buffer(comm->release_data(std::move(fut)));
            return std::move(chunk);
        }
    );
    std::erase(chunks, nullptr);
    return result;
}

}  // namespace rapidsmpf::coll::detail
