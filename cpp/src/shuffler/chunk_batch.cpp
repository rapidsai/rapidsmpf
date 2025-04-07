/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>

#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/chunk_batch.hpp>

namespace rapidsmp::shuffler::detail {

ChunkBatch::ChunkBatch(
    std::unique_ptr<std::vector<uint8_t>> metadata, std::unique_ptr<Buffer> payload_data
)
    : metadata_buffer_(std::move(metadata)), payload_data_(std::move(payload_data)) {}

Rank ChunkBatch::destination() const {
    // Maybe converted to constexpr in C++20
    return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data())->dest_rank;
}

size_t ChunkBatch::size() const {
    // Maybe converted to constexpr in C++20
    return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data())->num_chunks;
}

uint32_t ChunkBatch::id() const {
    // Maybe converted to constexpr in C++20
    return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data())->id;
}

ChunkBatch ChunkBatch::create(
    uint32_t id,
    Rank dest_rank,
    std::vector<Chunk>&& chunks,
    BufferResource* /* br */,
    rmm::cuda_stream_view /* stream */
) {
    ChunkBatch batch;

    // accumulate the metadata and payload sizes. Metadata buffer also contains the
    // metadata header information as well. Therefore, initialize value for len(chunks) *
    // sizeof(metadata header)
    size_t batch_metadata_size =
        sizeof(BatchHeader) + sizeof(Chunk::MetadataMessageHeader) * chunks.size();
    size_t batch_payload_size = 0;
    MemoryType mem_type = chunks[0].gpu_data->mem_type;
    for (auto& chunk : chunks) {
        batch_metadata_size += chunk.metadata->size();
        batch_payload_size += chunk.gpu_data->size;

        RAPIDSMP_EXPECTS(
            mem_type == chunk.gpu_data->mem_type,
            "All chunks in a batch should be of the same memory type"
        );
    }

    batch.metadata_buffer_ = std::make_unique<std::vector<uint8_t>>(batch_metadata_size);

    // inject header information to the metadata buffer
    auto header = reinterpret_cast<BatchHeader*>(batch.metadata_buffer_->data());
    header->id = id;
    header->dest_rank = dest_rank;
    header->num_chunks = chunks.size();

    if (chunks.empty()) {
        return batch;
    }

    size_t metadata_offset = sizeof(BatchHeader);  // skip the header
    for (auto&& chunk : chunks) {
        // copy metadata
        chunk.to_metadata_message(*batch.metadata_buffer_, metadata_offset);
        size_t chunk_metadata_size = chunk.metadata ? chunk.metadata->size() : 0;
        metadata_offset += (sizeof(Chunk::MetadataMessageHeader) + chunk_metadata_size);

        // TODO: copy payload data
    }

    return batch;
}

ChunkBatch ChunkBatch::create(
    std::unique_ptr<std::vector<uint8_t>> metadata, std::unique_ptr<Buffer> payload_data
) {
    RAPIDSMP_EXPECTS(metadata, "metadata buffer is null");
    RAPIDSMP_EXPECTS(payload_data, "payload buffer is null");
    RAPIDSMP_EXPECTS(
        metadata->size() >= sizeof(BatchHeader),
        "metadata buffer size is less than the header size"
    );
    ChunkBatch batch{std::move(metadata), std::move(payload_data)};
    return batch;
}

bool rapidsmp::shuffler::detail::ChunkBatchMetadataReader::has_next() const {
    return current_idx_ < batch_.size();
}

Chunk rapidsmp::shuffler::detail::ChunkBatchMetadataReader::next() {
    auto chunk_header = reinterpret_cast<Chunk::MetadataMessageHeader const*>(
        batch_.metadata_buffer_->data() + metadata_offset_
    );
    metadata_offset_ += sizeof(Chunk::MetadataMessageHeader);

    // create a buffer to copy the metadata
    auto chunk_metadata =
        std::make_unique<std::vector<uint8_t>>(chunk_header->metadata_size);
    // copy the metadata from parent buffer
    std::memcpy(
        chunk_metadata->data(),
        batch_.metadata_buffer_->data() + metadata_offset_,
        chunk_header->metadata_size
    );
    metadata_offset_ += chunk_header->metadata_size;

    return {
        chunk_header->pid,
        chunk_header->cid,
        chunk_header->expected_num_chunks,
        chunk_header->gpu_data_size,
        std::move(chunk_metadata),
        nullptr,  // payload buffer
    };
}

}  // namespace rapidsmp::shuffler::detail