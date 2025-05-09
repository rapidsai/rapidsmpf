/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {

Chunk::Chunk(
    ChunkID chunk_id,
    size_t n_messages,
    std::vector<PartID> part_ids,
    std::vector<size_t> expected_num_chunks,
    std::vector<uint32_t> meta_offsets,
    std::vector<uint64_t> data_offsets,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : chunk_id_{chunk_id},
      n_messages_{n_messages},
      part_ids_{std::move(part_ids)},
      expected_num_chunks_{std::move(expected_num_chunks)},
      meta_offsets_{std::move(meta_offsets)},
      data_offsets_{std::move(data_offsets)},
      metadata_{std::move(metadata)},
      data_{std::move(data)} {}

Chunk Chunk::get_data(
    ChunkID new_chunk_id, size_t i, rmm::cuda_stream_view /* stream */
) {
    RAPIDSMPF_EXPECTS(i < n_messages(), "index out of bounds", std::out_of_range);

    if (is_control_message(i)) {
        return from_finished_partition(new_chunk_id, part_id(i), expected_num_chunks(i));
    }

    if (n_messages() == 1) {  // i == 0, already verified
        // If there is only one message, move the metadata and data to the new chunk.
        return Chunk(
            new_chunk_id,
            1,
            {part_ids_[0]},
            {expected_num_chunks_[0]},
            {meta_offsets_[0]},
            {data_offsets_[0]},
            std::move(metadata_),
            std::move(data_)
        );
    } else {
        RAPIDSMPF_EXPECTS(false, "not implemented");
        // TODO: slice and copy data
    }

    return {new_chunk_id, 1, {}, {}, {}, {}};  // never reached
}

Chunk Chunk::from_packed_data(
    ChunkID chunk_id,
    PartID part_id,
    PackedData&& packed_data,
    std::shared_ptr<Buffer::Event> event,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    std::vector<uint32_t> meta_offsets{0};
    if (packed_data.metadata) {
        meta_offsets[0] = static_cast<uint32_t>(packed_data.metadata->size());
    }

    std::vector<uint64_t> data_offsets{0};
    if (packed_data.gpu_data) {
        data_offsets[0] = packed_data.gpu_data->size();
    }

    return {
        chunk_id,
        1,
        {part_id},
        {0},  // expected_num_chunks
        std::move(meta_offsets),
        std::move(data_offsets),
        std::move(packed_data.metadata),
        packed_data.gpu_data
            ? br->move(std::move(packed_data.gpu_data), stream, std::move(event))
            : nullptr
    };
}

Chunk Chunk::from_finished_partition(
    ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
) {
    return {chunk_id, 1, {part_id}, {expected_num_chunks}, {0}, {0}};
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
        n_messages,
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
    size_t n_messages =
        *reinterpret_cast<size_t const*>(serialized_buf.data() + sizeof(ChunkID));

    if (n_messages == 0) {  // no messages
        return false;
    }

    // Check if buffer is large enough to contain all the messages' metadata
    size_t header_size = metadata_message_header_size(n_messages);
    if (serialized_buf.size() < header_size) {
        return false;
    }

    // For each message, validate the metadata and data sizes
    auto const* psum_meta = reinterpret_cast<uint32_t const*>(
        serialized_buf.data() + sizeof(ChunkID) + sizeof(size_t)
        + n_messages * (sizeof(PartID) + sizeof(size_t))
    );
    auto const* psum_data = reinterpret_cast<uint64_t const*>(psum_meta + n_messages);

    // Check if prefix sums are non-decreasing
    bool is_non_decreasing = true;
    for (size_t i = 1; i < n_messages; ++i) {
        is_non_decreasing &= (psum_meta[i] >= psum_meta[i - 1]);
        is_non_decreasing &= (psum_data[i] >= psum_data[i - 1]);
    }
    if (!is_non_decreasing) {
        return false;
    }

    // Check if the total metadata size matches the buffer size
    size_t total_meta_size = psum_meta[n_messages - 1];
    if (serialized_buf.size() != header_size + total_meta_size) {
        return false;
    }

    return true;
}

std::string Chunk::str(std::size_t /*max_nbytes*/, rmm::cuda_stream_view /*stream*/)
    const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id() << ", n=" << n_messages() << ", ";

    // Add message details
    for (size_t i = 0; i < n_messages(); ++i) {
        ss << "msg[" << i << "]={";
        ss << "part_id=" << part_ids_[i];
        ss << ", expected_num_chunks=" << expected_num_chunks_[i];
        ss << ", metadata_size=" << (meta_offsets_.empty() ? 0 : meta_offsets_[i]);
        ss << ", data_size=" << (data_offsets_.empty() ? 0 : data_offsets_[i]);
        ss << "},";
    }
    ss << ")";
    return ss.str();
}

std::unique_ptr<std::vector<uint8_t>> Chunk::serialize() const {
    size_t metadata_buf_size =
        metadata_message_header_size(n_messages_) + (metadata_ ? metadata_->size() : 0);
    auto metadata_buf = std::make_unique<std::vector<uint8_t>>(metadata_buf_size);

    uint8_t* p = metadata_buf->data();
    // Write chunk ID
    std::memcpy(p, &chunk_id_, sizeof(ChunkID));
    p += sizeof(ChunkID);

    // Write number of messages
    std::memcpy(p, &n_messages_, sizeof(size_t));
    p += sizeof(size_t);

    // Write partition IDs
    std::memcpy(p, part_ids_.data(), n_messages_ * sizeof(PartID));
    p += n_messages_ * sizeof(PartID);

    // Write expected number of chunks
    std::memcpy(p, expected_num_chunks_.data(), n_messages_ * sizeof(size_t));
    p += n_messages_ * sizeof(size_t);

    // Write metadata offsets
    std::memcpy(p, meta_offsets_.data(), n_messages_ * sizeof(uint32_t));
    p += n_messages_ * sizeof(uint32_t);

    // Write data offsets
    std::memcpy(p, data_offsets_.data(), n_messages_ * sizeof(uint64_t));
    p += n_messages_ * sizeof(uint64_t);

    // Write concatenated metadata
    if (metadata_) {
        std::memcpy(p, metadata_->data(), metadata_->size());
        metadata_->clear();
    }

    return metadata_buf;
}


}  // namespace rapidsmpf::shuffler::detail
