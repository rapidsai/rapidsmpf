/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk_batch.hpp>

namespace rapidsmpf::shuffler::detail {


ChunkBatch::ChunkBatch(
    std::unique_ptr<std::vector<uint8_t>> metadata, std::unique_ptr<Buffer> payload_data
)
    : metadata_buffer_(std::move(metadata)), payload_data_(std::move(payload_data)) {}

ChunkBatch ChunkBatch::create(
    uint32_t id,
    Rank dest_rank,
    std::vector<Chunk>&& chunks,
    BufferResource* br,
    rmm::cuda_stream_view stream
) {
    // accumulate the metadata and payload sizes. Metadata buffer also contains the
    // metadata header information as well. Therefore, initialize value for len(chunks) *
    // sizeof(metadata header)
    size_t batch_metadata_size =
        batch_header_size + chunk_metadata_header_size * chunks.size();
    size_t batch_payload_size = 0;

    // create the initial metadata buffer
    auto metadata_buffer = std::make_unique<std::vector<uint8_t>>(batch_metadata_size);

    // inject header information to the metadata buffer
    auto header = reinterpret_cast<BatchHeader*>(metadata_buffer->data());
    header->id = id;
    header->dest_rank = dest_rank;
    header->num_chunks = chunks.size();

    if (chunks.empty()) {  // preemptively return if there are no chunks
        return {std::move(metadata_buffer), nullptr};
    }

    // first traverse the chunks to calculate the metadata and payload sizes, and the
    // memory type of data buffers
    MemoryType mem_type;
    int num_payload_chunks = 0;
    for (const auto& chunk : chunks) {
        if (chunk.metadata) {
            batch_metadata_size += chunk.metadata->size();
        }

        if (chunk.gpu_data) {
            batch_payload_size += chunk.gpu_data->size;

            if (++num_payload_chunks == 1) {
                // if this was the first chunk with payloads, set mem type
                mem_type = chunk.gpu_data->mem_type();
            } else {  // num_payload_chunks > 1
                // TODO: add a policy to handle multiple types in the vector
                RAPIDSMPF_EXPECTS(
                    mem_type == chunk.gpu_data->mem_type(),
                    "All chunks in a batch should be of the same memory type"
                );
            }
        }
    }

    // resize the metadata buffer to the actual size
    metadata_buffer->resize(batch_metadata_size);

    // create payload data buffer
    std::unique_ptr<Buffer> payload_data;
    // allocate a separate data buffer, if there are >1 payloads
    if (batch_payload_size > 0 and num_payload_chunks > 1) {
        auto [reservation, _] = br->reserve(mem_type, batch_payload_size, false);
        RAPIDSMPF_EXPECTS(
            reservation.size() == batch_payload_size,
            "unable to reserve gpu memory for batch"
        );
        payload_data = br->allocate(mem_type, batch_payload_size, stream, reservation);
        RAPIDSMPF_EXPECTS(reservation.size() == 0, "didn't use all of the reservation");
    }

    // Now, traverse the chunks again, and copy data into buffers
    std::ptrdiff_t metadata_offset = batch_header_size;  // skip the header
    std::ptrdiff_t payload_offset = 0;
    for (auto&& chunk : chunks) {
        auto moved_chunk = std::move(chunk);  // move the chunk
        // copy metadata
        metadata_offset +=
            moved_chunk.to_metadata_message(*metadata_buffer, metadata_offset);

        // copy payload data
        if (moved_chunk.gpu_data) {
            if (num_payload_chunks == 1) {
                // this is the only chunk with payload, so move the data buffer
                payload_data = std::move(moved_chunk.gpu_data);
            } else {  // several payload chunks, so copy data to the payload buffer
                payload_offset +=
                    moved_chunk.gpu_data->copy_to(*payload_data, payload_offset, stream);
            }
        }
    }

    return {std::move(metadata_buffer), std::move(payload_data)};
}

ChunkBatch ChunkBatch::create(
    std::unique_ptr<std::vector<uint8_t>> metadata, std::unique_ptr<Buffer> payload_data
) {
    RAPIDSMPF_EXPECTS(metadata, "metadata buffer is null");
    RAPIDSMPF_EXPECTS(
        metadata->size() >= batch_header_size,
        "metadata buffer size is less than the header size"
    );
    // first create a chunk batch
    ChunkBatch batch{std::move(metadata), std::move(payload_data)};

    // visit chunk data and verify if the given buffers adhere to the format
    size_t visited_metadata_size = batch_header_size;
    size_t visited_payload_size = 0;
    batch.visit_chunk_data([&](Chunk::MetadataMessageHeader const* chunk_header,
                               auto const& /* metadata_buf */,
                               auto /* metadata_offset */,
                               auto const& /* payload_buf */,
                               auto /* payload_offset */) {
        visited_metadata_size +=
            (chunk_metadata_header_size + chunk_header->metadata_size);
        visited_payload_size += chunk_header->gpu_data_size;
    });
    RAPIDSMPF_EXPECTS(
        visited_metadata_size == batch.metadata_buffer_->size(),
        "visited metadata size doesn't match the metadata buffer size"
    );
    if (batch.payload_data_) {
        RAPIDSMPF_EXPECTS(
            visited_payload_size == batch.payload_data_->size,
            "visited payload size doesn't match the payload buffer size"
        );
    }
    return batch;
}

ChunkForwardIterator ChunkBatch::begin(rmm::cuda_stream_view stream) const {
    if (!cached_begin_iter_) {
        cached_begin_iter_ = std::unique_ptr<ChunkForwardIterator>(
            new ChunkForwardIterator(*this, batch_header_size, 0, stream)
        );
    }

    return *cached_begin_iter_;
}

ChunkForwardIterator ChunkBatch::end(rmm::cuda_stream_view stream) const {
    // this is a trivial operation because metadata_offset == metadata size() and no
    // chunk will be allocated inside the iterator.
    return ChunkForwardIterator(
        *this,
        std::ptrdiff_t(metadata_buffer_->size()),
        payload_data_ ? std::ptrdiff_t(payload_data_->size) : 0,
        stream
    );
}

ChunkForwardIterator::ChunkForwardIterator(
    ChunkBatch const& batch,
    std::ptrdiff_t metadata_offset,
    std::ptrdiff_t payload_offset,
    rmm::cuda_stream_view stream
)
    : batch_(batch),
      metadata_offset_(metadata_offset),
      payload_offset_(payload_offset),
      stream_(stream) {
    if (has_chunk()) {
        chunk_ = make_chunk(batch_, metadata_offset_, payload_offset_, stream_);
    }
}

ChunkForwardIterator& ChunkForwardIterator::operator++() {
    advance_chunk();
    return *this;
}

ChunkForwardIterator ChunkForwardIterator::operator++(int) {
    ChunkForwardIterator temp = *this;
    advance_chunk();
    return temp;
}

void ChunkForwardIterator::advance_chunk() {
    auto const* header = chunk_header();  // get the current header
    // skip to the next offset boundary
    metadata_offset_ +=
        std::ptrdiff_t(ChunkBatch::chunk_metadata_header_size + header->metadata_size);
    // skip to the next payload boundary
    payload_offset_ += std::ptrdiff_t(header->gpu_data_size);

    if (has_chunk()) {
        chunk_ = make_chunk(batch_, metadata_offset_, payload_offset_, stream_);
    } else {
        chunk_ = {};
    }
}

std::shared_ptr<Chunk> ChunkForwardIterator::make_chunk(
    ChunkBatch const& batch,
    std::ptrdiff_t metadata_offset,
    std::ptrdiff_t payload_offset,
    rmm::cuda_stream_view stream
) {
    auto chunk_header = reinterpret_cast<Chunk::MetadataMessageHeader const*>(
        batch.metadata_buffer_->data() + metadata_offset
    );
    metadata_offset += ChunkBatch::chunk_metadata_header_size;

    auto const& metadata_buf = *batch.metadata_buffer_;
    auto const& payload_buf = *batch.payload_data_;

    std::unique_ptr<std::vector<uint8_t>> chunk_metadata;
    if (chunk_header->metadata_size > 0) {  // only allocate if non-zero
        chunk_metadata = std::make_unique<std::vector<uint8_t>>(
            metadata_buf.begin() + metadata_offset,
            metadata_buf.begin() + metadata_offset
                + std::ptrdiff_t(chunk_header->metadata_size)
        );
    }

    std::unique_ptr<Buffer> chunk_payload;
    if (chunk_header->gpu_data_size > 0) {  // only copy slice if non-zero
        chunk_payload = payload_buf.copy_slice(
            payload_offset, std::ptrdiff_t(chunk_header->gpu_data_size), stream
        );
    }

    return std::make_shared<Chunk>(
        chunk_header->pid,
        chunk_header->cid,
        chunk_header->expected_num_chunks,
        chunk_header->gpu_data_size,
        std::move(chunk_metadata),
        std::move(chunk_payload)
    );
}

}  // namespace rapidsmpf::shuffler::detail
