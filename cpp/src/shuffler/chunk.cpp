/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler::detail {

Chunk::Chunk(
    ChunkID chunk_id,
    std::vector<PartID> part_ids,
    std::vector<size_t> expected_num_chunks,
    std::vector<uint32_t> meta_offsets,
    std::vector<uint64_t> data_offsets,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : chunk_id_{chunk_id},
      part_ids_{std::move(part_ids)},
      expected_num_chunks_{std::move(expected_num_chunks)},
      meta_offsets_{std::move(meta_offsets)},
      data_offsets_{std::move(data_offsets)},
      metadata_{std::move(metadata)},
      data_{std::move(data)} {
    RAPIDSMPF_EXPECTS(
        (part_ids_.size() > 0) && (part_ids_.size() == expected_num_chunks_.size())
            && (part_ids_.size() == meta_offsets_.size())
            && (part_ids_.size() == data_offsets_.size()),
        "invalid chunk: input vectors have different sizes"
    );
    RAPIDSMPF_EXPECTS(part_ids_.size() == 1, "multi-message chunks are not supported");
}

Chunk Chunk::get_data(ChunkID new_chunk_id, BufferResource* br) {
    RAPIDSMPF_EXPECTS(n_messages() == 1, "multi-message chunks are not supported");

    if (is_control_message()) {
        return from_finished_partition(new_chunk_id, part_id(), expected_num_chunks());
    }
    auto stream = br->stream_pool().get_stream();

    // Single-message chunk - move the metadata and data to the new chunk
    return Chunk(
        new_chunk_id,
        {part_ids_[0]},
        {expected_num_chunks_[0]},
        {meta_offsets_[0]},
        {data_offsets_[0]},
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
        {part_id},
        {0},  // expected_num_chunks
        {static_cast<uint32_t>(packed_data.metadata->size())},
        {packed_data.data->size},
        std::move(packed_data.metadata),
        std::move(packed_data.data),
    };
}

Chunk Chunk::from_finished_partition(
    ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
) {
    return {chunk_id, {part_id}, {expected_num_chunks}, {0}, {0}};
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

    size_t n_messages;
    std::memcpy(&n_messages, msg.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    std::vector<PartID> part_ids(n_messages);
    std::memcpy(part_ids.data(), msg.data() + offset, n_messages * sizeof(PartID));
    offset += n_messages * sizeof(PartID);

    std::vector<size_t> expected_num_chunks(n_messages);
    std::memcpy(
        expected_num_chunks.data(), msg.data() + offset, n_messages * sizeof(size_t)
    );
    offset += n_messages * sizeof(size_t);

    std::vector<uint32_t> meta_offsets(n_messages);
    std::memcpy(meta_offsets.data(), msg.data() + offset, n_messages * sizeof(uint32_t));
    offset += n_messages * sizeof(uint32_t);

    std::vector<uint64_t> data_offsets(n_messages);
    std::memcpy(data_offsets.data(), msg.data() + offset, n_messages * sizeof(uint64_t));
    offset += n_messages * sizeof(uint64_t);

    auto concat_metadata = std::make_unique<std::vector<uint8_t>>(
        msg.begin() + static_cast<int64_t>(offset), msg.end()
    );

    return {
        chunk_id,
        std::move(part_ids),
        std::move(expected_num_chunks),
        std::move(meta_offsets),
        std::move(data_offsets),
        std::move(concat_metadata),
        nullptr
    };
}

bool Chunk::validate_format(std::vector<uint8_t> const& serialized_buf) {
    // Check if buffer is large enough to contain at least the header
    if (serialized_buf.size() < sizeof(ChunkID) + sizeof(size_t)) {
        return false;
    }

    // Get number of messages
    size_t n = 0;
    std::memcpy(&n, serialized_buf.data() + sizeof(ChunkID), sizeof(size_t));

    if (n == 0) {  // no messages
        return false;
    }

    // Check if buffer is large enough to contain all the messages' metadata
    size_t header_size = metadata_message_header_size(n);
    if (serialized_buf.size() < header_size) {
        return false;
    }

    // Check if the partition IDs are unique
    std::unordered_set<PartID> seen_pids;
    seen_pids.reserve(n);
    auto const* pids = serialized_buf.data() + sizeof(ChunkID) + sizeof(size_t);
    for (size_t i = 0; i < n; ++i) {
        PartID pid;
        std::memcpy(&pid, pids + i * sizeof(PartID), sizeof(PartID));
        if (!seen_pids.emplace(pid).second) {
            return false;
        }
    }

    // For each message, validate the metadata and data sizes
    uint8_t const* meta_offsets_start =
        serialized_buf.data()
        + (sizeof(ChunkID) + sizeof(size_t) + n * (sizeof(PartID) + sizeof(size_t)));
    uint8_t const* data_offsets_start = meta_offsets_start + n * sizeof(uint32_t);

    // Check if prefix sums are non-decreasing
    bool is_non_decreasing = true;
    for (size_t i = 1; i < n; ++i) {
        uint32_t prev_meta_offset, this_meta_offset;
        std::memcpy(
            &prev_meta_offset,
            meta_offsets_start + (i - 1) * sizeof(uint32_t),
            sizeof(uint32_t)
        );
        std::memcpy(
            &this_meta_offset, meta_offsets_start + i * sizeof(uint32_t), sizeof(uint32_t)
        );
        is_non_decreasing &= (this_meta_offset >= prev_meta_offset);

        uint64_t prev_data_offset, this_data_offset;
        std::memcpy(
            &prev_data_offset,
            data_offsets_start + (i - 1) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
        std::memcpy(
            &this_data_offset, data_offsets_start + i * sizeof(uint64_t), sizeof(uint64_t)
        );
        is_non_decreasing &= (this_data_offset >= prev_data_offset);
    }
    if (!is_non_decreasing) {
        return false;
    }

    // Check if the total metadata size matches the buffer size
    uint32_t total_meta_size;
    std::memcpy(
        &total_meta_size,
        meta_offsets_start + (n - 1) * sizeof(uint32_t),
        sizeof(uint32_t)
    );
    if (serialized_buf.size() != header_size + total_meta_size) {
        return false;
    }

    return true;
}

std::string Chunk::str() const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id();
    ss << ", part_id=" << part_ids_[0];
    ss << ", expected_num_chunks=" << expected_num_chunks_[0];
    ss << ", metadata_size=" << (meta_offsets_.empty() ? 0 : meta_offsets_[0]);
    ss << ", data_size=" << (data_offsets_.empty() ? 0 : data_offsets_[0]);
    ss << ")";
    return ss.str();
}

std::unique_ptr<std::vector<uint8_t>> Chunk::serialize() const {
    size_t n = this->n_messages();

    size_t metadata_buf_size =
        metadata_message_header_size(n) + (metadata_ ? metadata_->size() : 0);
    auto metadata_buf = std::make_unique<std::vector<uint8_t>>(metadata_buf_size);

    uint8_t* p = metadata_buf->data();
    // Write chunk ID
    std::memcpy(p, &chunk_id_, sizeof(ChunkID));
    p += sizeof(ChunkID);

    // Write number of messages
    std::memcpy(p, &n, sizeof(size_t));
    p += sizeof(size_t);

    // Write partition IDs
    std::memcpy(p, part_ids_.data(), n * sizeof(PartID));
    p += n * sizeof(PartID);

    // Write expected number of chunks
    std::memcpy(p, expected_num_chunks_.data(), n * sizeof(size_t));
    p += n * sizeof(size_t);

    // Write metadata offsets
    std::memcpy(p, meta_offsets_.data(), n * sizeof(uint32_t));
    p += n * sizeof(uint32_t);

    // Write data offsets
    std::memcpy(p, data_offsets_.data(), n * sizeof(uint64_t));
    p += n * sizeof(uint64_t);

    // Write concatenated metadata
    if (metadata_) {
        std::memcpy(p, metadata_->data(), metadata_->size());
        metadata_->clear();
    }

    return metadata_buf;
}

}  // namespace rapidsmpf::shuffler::detail
