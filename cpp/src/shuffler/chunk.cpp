/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstring>
#include <sstream>
#include <utility>

#include <cuda/stream>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler::detail {

Chunk::Chunk(
    ChunkID chunk_id,
    PartID part_id,
    std::size_t expected_num_chunks,
    std::uint32_t metadata_size,
    std::uint64_t data_size,
    std::unique_ptr<std::vector<std::uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : chunk_id_{chunk_id},
      part_id_{part_id},
      expected_num_chunks_{expected_num_chunks},
      metadata_size_{metadata_size},
      data_size_{data_size},
      metadata_{std::move(metadata)},
      data_{std::move(data)} {}

Chunk Chunk::from_packed_data(
    ChunkID chunk_id, PartID part_id, PackedData&& packed_data
) {
    RAPIDSMPF_EXPECTS(packed_data.metadata != nullptr, "packed_data.metadata is nullptr");
    RAPIDSMPF_EXPECTS(packed_data.data != nullptr, "packed_data.data is nullptr");
    return Chunk{
        chunk_id,
        part_id,
        0,  // expected_num_chunks
        static_cast<std::uint32_t>(packed_data.metadata->size()),
        packed_data.data->size,
        std::move(packed_data.metadata),
        std::move(packed_data.data),
    };
}

Chunk Chunk::from_finished_partition(
    ChunkID chunk_id, PartID part_id, std::size_t expected_num_chunks
) {
    return {chunk_id, part_id, expected_num_chunks, 0, 0};
}

Chunk Chunk::deserialize(
    std::vector<std::uint8_t> const& msg,
    BufferResource* br,
    bool validate,
    std::unique_ptr<Buffer> data
) {
    if (validate) {
        RAPIDSMPF_EXPECTS(
            validate_format(msg), "serialized message does not follow the expected format"
        );
    }
    std::size_t offset = 0;

    ChunkID chunk_id;
    std::memcpy(&chunk_id, msg.data() + offset, sizeof(ChunkID));
    offset += sizeof(ChunkID);

    PartID part_id;
    std::memcpy(&part_id, msg.data() + offset, sizeof(PartID));
    offset += sizeof(PartID);

    std::size_t expected_num_chunks;
    std::memcpy(&expected_num_chunks, msg.data() + offset, sizeof(std::size_t));
    offset += sizeof(std::size_t);

    std::uint32_t metadata_size;
    std::memcpy(&metadata_size, msg.data() + offset, sizeof(std::uint32_t));
    offset += sizeof(std::uint32_t);

    std::uint64_t data_size;
    std::memcpy(&data_size, msg.data() + offset, sizeof(std::uint64_t));
    offset += sizeof(std::uint64_t);

    auto concat_metadata = std::make_unique<std::vector<std::uint8_t>>(
        msg.begin() + safe_cast<std::int64_t>(offset), msg.end()
    );

    if (!data && expected_num_chunks == 0) {
        RAPIDSMPF_EXPECTS(
            br != nullptr, "Deserializing non-control Chunk requires a BufferResource"
        );
        auto reservation = br->try_reserve_or_spill(data_size, MEMORY_TYPES);
        RAPIDSMPF_EXPECTS(
            reservation.has_value(),
            "failed to reserve receive buffer after spilling",
            std::runtime_error
        );
        data = br->make_buffer(br->stream_pool()->get_stream(), std::move(*reservation));
        if (rapidsmpf::contains(SPILL_TARGET_MEMORY_TYPES, data->mem_type())) {
            br->statistics()->add_bytes_stat("recv-into-host-memory", data_size);
        }
    }

    return {
        chunk_id,
        part_id,
        expected_num_chunks,
        metadata_size,
        data_size,
        std::move(concat_metadata),
        std::move(data)
    };
}

bool Chunk::validate_format(std::vector<std::uint8_t> const& serialized_buf) {
    // Check if buffer is large enough to contain at least the header
    constexpr std::size_t header_size = metadata_message_header_size();
    if (serialized_buf.size() < header_size) {
        return false;
    }

    // Read metadata_size from the header
    std::uint8_t const* sizes_start =
        serialized_buf.data() + sizeof(ChunkID) + sizeof(PartID) + sizeof(std::size_t);

    std::uint32_t metadata_size;
    std::memcpy(&metadata_size, sizes_start, sizeof(std::uint32_t));

    // Check if the total metadata size matches the buffer size
    if (serialized_buf.size() != header_size + metadata_size) {
        return false;
    }

    return true;
}

std::filesystem::path const& Chunk::disk_path() const {
    RAPIDSMPF_EXPECTS(disk_data_, "chunk is not disk-resident");
    return disk_data_->path();
}

void Chunk::spill_from_device(BufferResource& br) {
    if (data_size_ == 0 || disk_data_ || !data_) {
        return;
    }
    if (data_->mem_type() != MemoryType::DEVICE) {
        return;
    }
    if (auto reservation = br.try_reserve(data_size_, SPILL_TARGET_MEMORY_TYPES)) {
        data_ = br.move(std::move(data_), *reservation);
        return;
    }
    spill_to_disk(br);
}

void Chunk::spill_to_disk(BufferResource& br) {
    if (data_size_ == 0) {
        return;
    }
    RAPIDSMPF_EXPECTS(
        data_ && !disk_data_, "spill_to_disk requires an exclusive in-memory payload"
    );
    data_->latest_write_event().host_wait();
    disk_data_ = disk::DiskBuffer::from_buffer(std::move(data_), br);
}

void Chunk::restore_from_disk(BufferResource& br) {
    RAPIDSMPF_EXPECTS(
        disk_data_ && !data_, "restore_from_disk requires an exclusive disk payload"
    );
    auto const size = disk_data_->size();
    constexpr std::array restore_mem_types{
        MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST
    };
    auto reservation = br.try_reserve_or_spill(size, restore_mem_types);
    RAPIDSMPF_EXPECTS(
        reservation.has_value(),
        "failed to reserve memory to restore a disk-resident chunk after spilling",
        std::runtime_error
    );
    data_ = disk::DiskBuffer::restore(
        std::move(disk_data_), *reservation, br.stream_pool()->get_stream()
    );
}

std::string Chunk::str() const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id();
    ss << ", part_id=" << part_id_;
    ss << ", expected_num_chunks=" << expected_num_chunks_;
    ss << ", metadata_size=" << metadata_size_;
    ss << ", data_size=" << data_size_;
    if (disk_data_) {
        ss << ", on_disk";
    }
    ss << ")";
    return ss.str();
}

std::unique_ptr<std::vector<std::uint8_t>> Chunk::serialize() const {
    std::size_t metadata_buf_size =
        metadata_message_header_size() + (metadata_ ? metadata_->size() : 0);
    auto metadata_buf = std::make_unique<std::vector<std::uint8_t>>(metadata_buf_size);

    std::uint8_t* p = metadata_buf->data();
    // Write chunk ID
    std::memcpy(p, &chunk_id_, sizeof(ChunkID));
    p += sizeof(ChunkID);

    // Write partition ID
    std::memcpy(p, &part_id_, sizeof(PartID));
    p += sizeof(PartID);

    // Write expected number of chunks
    std::memcpy(p, &expected_num_chunks_, sizeof(std::size_t));
    p += sizeof(std::size_t);

    // Write metadata offset (size)
    std::memcpy(p, &metadata_size_, sizeof(std::uint32_t));
    p += sizeof(std::uint32_t);

    // Write data offset (size)
    std::memcpy(p, &data_size_, sizeof(std::uint64_t));
    p += sizeof(std::uint64_t);

    // Write concatenated metadata
    if (metadata_) {
        std::memcpy(p, metadata_->data(), metadata_->size());
        metadata_->clear();
    }

    return metadata_buf;
}

std::ostream& operator<<(std::ostream& os, Chunk const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
