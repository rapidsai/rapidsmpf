/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <sstream>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler::detail {

Chunk::Chunk(
    ChunkID chunk_id,
    PartID part_id,
    size_t expected_num_chunks,
    uint32_t metadata_size,
    uint64_t data_size,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : chunk_id_{chunk_id},
      part_id_{part_id},
      expected_num_chunks_{expected_num_chunks},
      metadata_size_{metadata_size},
      data_size_{data_size},
      metadata_{std::move(metadata)},
      data_{std::move(data)} {}

Chunk Chunk::get_data(ChunkID new_chunk_id, BufferResource* br) {
    if (is_control_message()) {
        return from_finished_partition(new_chunk_id, part_id(), expected_num_chunks());
    }
    auto stream = br->stream_pool().get_stream();

    return Chunk(
        new_chunk_id,
        part_id_,
        expected_num_chunks_,
        metadata_size_,
        data_size_,
        std::move(metadata_),
        data_ ? std::move(data_)
              : br->allocate(stream, br->reserve_or_fail(0, MemoryType::HOST))
    );
}

Chunk Chunk::from_packed_data(
    ChunkID chunk_id, PartID part_id, PackedData&& packed_data
) {
    RAPIDSMPF_EXPECTS(packed_data.metadata != nullptr, "packed_data.metadata is nullptr");
    RAPIDSMPF_EXPECTS(packed_data.data != nullptr, "packed_data.data is nullptr");
    return Chunk{
        chunk_id,
        part_id,
        0,  // expected_num_chunks
        static_cast<uint32_t>(packed_data.metadata->size()),
        packed_data.data->size,
        std::move(packed_data.metadata),
        std::move(packed_data.data),
    };
}

Chunk Chunk::from_finished_partition(
    ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
) {
    return {chunk_id, part_id, expected_num_chunks, 0, 0};
}

Chunk Chunk::deserialize(std::vector<uint8_t> const& msg, bool validate) {
    if (validate) {
        RAPIDSMPF_EXPECTS(
            validate_format(msg), "serialized message does not follow the expected format"
        );
    }
    size_t offset = 0;

    ChunkID chunk_id;
    std::memcpy(&chunk_id, msg.data() + offset, sizeof(ChunkID));
    offset += sizeof(ChunkID);

    PartID part_id;
    std::memcpy(&part_id, msg.data() + offset, sizeof(PartID));
    offset += sizeof(PartID);

    size_t expected_num_chunks;
    std::memcpy(&expected_num_chunks, msg.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    uint32_t metadata_size;
    std::memcpy(&metadata_size, msg.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    uint64_t data_size;
    std::memcpy(&data_size, msg.data() + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    auto concat_metadata = std::make_unique<std::vector<uint8_t>>(
        msg.begin() + static_cast<int64_t>(offset), msg.end()
    );

    return {
        chunk_id,
        part_id,
        expected_num_chunks,
        metadata_size,
        data_size,
        std::move(concat_metadata),
        nullptr
    };
}

bool Chunk::validate_format(std::vector<uint8_t> const& serialized_buf) {
    // Check if buffer is large enough to contain at least the header
    constexpr size_t header_size = metadata_message_header_size();
    if (serialized_buf.size() < header_size) {
        return false;
    }

    // Read metadata_size from the header
    uint8_t const* sizes_start =
        serialized_buf.data() + sizeof(ChunkID) + sizeof(PartID) + sizeof(size_t);

    uint32_t metadata_size;
    std::memcpy(&metadata_size, sizes_start, sizeof(uint32_t));

    // Check if the total metadata size matches the buffer size
    if (serialized_buf.size() != header_size + metadata_size) {
        return false;
    }

    return true;
}

std::string Chunk::str() const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id();
    ss << ", part_id=" << part_id_;
    ss << ", expected_num_chunks=" << expected_num_chunks_;
    ss << ", metadata_size=" << metadata_size_;
    ss << ", data_size=" << data_size_;
    ss << ")";
    return ss.str();
}

std::unique_ptr<std::vector<uint8_t>> Chunk::serialize() const {
    size_t metadata_buf_size =
        metadata_message_header_size() + (metadata_ ? metadata_->size() : 0);
    auto metadata_buf = std::make_unique<std::vector<uint8_t>>(metadata_buf_size);

    uint8_t* p = metadata_buf->data();
    // Write chunk ID
    std::memcpy(p, &chunk_id_, sizeof(ChunkID));
    p += sizeof(ChunkID);

    // Write partition ID
    std::memcpy(p, &part_id_, sizeof(PartID));
    p += sizeof(PartID);

    // Write expected number of chunks
    std::memcpy(p, &expected_num_chunks_, sizeof(size_t));
    p += sizeof(size_t);

    // Write metadata offset (size)
    std::memcpy(p, &metadata_size_, sizeof(uint32_t));
    p += sizeof(uint32_t);

    // Write data offset (size)
    std::memcpy(p, &data_size_, sizeof(uint64_t));
    p += sizeof(uint64_t);

    // Write concatenated metadata
    if (metadata_) {
        std::memcpy(p, metadata_->data(), metadata_->size());
        metadata_->clear();
    }

    return metadata_buf;
}

std::unique_ptr<std::vector<uint8_t>> ReadyForDataMessage::pack() {
    auto msg = std::make_unique<std::vector<uint8_t>>(sizeof(ChunkID));
    std::memcpy(msg->data(), &cid, sizeof(cid));
    return msg;
}

ReadyForDataMessage ReadyForDataMessage::unpack(
    std::unique_ptr<std::vector<uint8_t>> const& msg
) {
    ChunkID cid;
    std::memcpy(&cid, msg->data(), sizeof(cid));
    return ReadyForDataMessage{cid};
}

std::string ReadyForDataMessage::str() const {
    std::stringstream ss;
    ss << "ReadyForDataMessage(cid=" << cid << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, Chunk const& obj) {
    os << obj.str();
    return os;
}

std::ostream& operator<<(std::ostream& os, ReadyForDataMessage const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
