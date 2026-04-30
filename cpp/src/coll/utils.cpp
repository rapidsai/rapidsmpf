/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

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
    std::vector<std::uint8_t>& data, BufferResource* br
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
    return std::unique_ptr<Chunk>(new Chunk(
        id,
        Chunk::INVALID_RANK,
        std::move(metadata),
        br->allocate(
            br->stream_pool().get_stream(), br->reserve_or_fail(data_size, MEMORY_TYPES)
        )
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

std::size_t PostBox::spill(BufferResource* br, std::size_t amount) {
    std::lock_guard lock(mutex_);
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
        auto reservation =
            br->reserve_or_fail(chunk->data_size(), SPILL_TARGET_MEMORY_TYPES);
        chunk->attach_data_buffer(br->move(chunk->release_data_buffer(), reservation));
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
        auto chunk = pos == spillable_chunks.end() ? spillable_chunks.back() : *pos;
        total_spilled += spill_chunk(chunk);
        if (total_spilled >= amount) {
            break;
        }
        spillable_chunks.pop_back();
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
