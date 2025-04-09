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
    std::optional<MemoryType> mem_type;
    for (const auto& chunk : chunks) {
        if (chunk.metadata) {
            batch_metadata_size += chunk.metadata->size();
        }

        if (chunk.gpu_data) {
            batch_payload_size += chunk.gpu_data->size;

            if (!mem_type) {
                mem_type = chunk.gpu_data->mem_type();
            } else {
                // TODO: add a policy to handle multiple types in the vector
                RAPIDSMP_EXPECTS(
                    *mem_type == chunk.gpu_data->mem_type(),
                    "All chunks in a batch should be of the same memory type"
                );
            }
        }
    }

    // resize the metadata buffer to the actual size
    metadata_buffer->resize(batch_metadata_size);

    // create payload data buffer
    std::unique_ptr<Buffer> payload_data;
    if (batch_payload_size > 0) {
        auto [reservation, _] = br->reserve(*mem_type, batch_payload_size, false);
        RAPIDSMP_EXPECTS(
            reservation.size() == batch_payload_size,
            "unable to reserve gpu memory for batch"
        );
        payload_data = br->allocate(*mem_type, batch_payload_size, stream, reservation);
        RAPIDSMP_EXPECTS(reservation.size() == 0, "didn't use all of the reservation");
    }

    // Now, traverse the chunks again, and copy data into buffers
    std::ptrdiff_t metadata_offset = batch_header_size;  // skip the header
    std::ptrdiff_t payload_offset = 0;
    for (auto&& chunk : chunks) {
        // copy metadata
        metadata_offset += chunk.to_metadata_message(*metadata_buffer, metadata_offset);

        // copy payload data
        if (chunk.gpu_data) {
            payload_offset +=
                chunk.gpu_data->copy_to(*payload_data, payload_offset, stream);
        }
    }

    return {std::move(metadata_buffer), std::move(payload_data)};
}

ChunkBatch ChunkBatch::create(
    std::unique_ptr<std::vector<uint8_t>> metadata, std::unique_ptr<Buffer> payload_data
) {
    RAPIDSMP_EXPECTS(metadata, "metadata buffer is null");
    RAPIDSMP_EXPECTS(
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
    RAPIDSMP_EXPECTS(
        visited_metadata_size == batch.metadata_buffer_->size(),
        "visited metadata size doesn't match the metadata buffer size"
    );
    if (batch.payload_data_) {
        RAPIDSMP_EXPECTS(
            visited_payload_size == batch.payload_data_->size,
            "visited payload size doesn't match the payload buffer size"
        );
    }
    return batch;
}

// bool rapidsmp::shuffler::detail::ChunkBatchMetadataReader::has_next() const {
//     return current_idx_ < batch_.size();
// }

// Chunk rapidsmp::shuffler::detail::ChunkBatchMetadataReader::next() {
//     auto chunk_header = reinterpret_cast<Chunk::MetadataMessageHeader const*>(
//         batch_.metadata_buffer_->data() + metadata_offset_
//     );
//     metadata_offset_ += sizeof(Chunk::MetadataMessageHeader);

//     // create a metadata buffer by slicing the parent metadata buffer
//     auto chunk_metadata = std::make_unique<std::vector<uint8_t>>(
//         batch_.metadata_buffer_->cbegin() + metadata_offset_,
//         batch_.metadata_buffer_->cbegin() + metadata_offset_ +
//         chunk_header->metadata_size
//     );
//     metadata_offset_ += chunk_header->metadata_size;

//     // create a payload buffer by slicing the parent payload buffer
//     auto chunk_payload = batch_.payload_data_->copy_slice(
//         stream_, payload_offset_, chunk_header->gpu_data_size
//     );
//     payload_offset_ += chunk_header->gpu_data_size;

//     return {
//         chunk_header->pid,
//         chunk_header->cid,
//         chunk_header->expected_num_chunks,
//         chunk_header->gpu_data_size,
//         std::move(chunk_metadata),
//         nullptr,  // payload buffer
//     };
// }

}  // namespace rapidsmp::shuffler::detail
