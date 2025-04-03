/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>

#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/chunk_batch.hpp>

namespace rapidsmp::shuffler::detail {

ChunkBatch::ChunkBatch(Rank dest_rank, size_t num_chunks)
    : dest_rank_(dest_rank), num_chunks_(num_chunks) {}

// void copy_host_to_buf(
//     void const* src,
//     size_t const size,
//     void* dest,
//     MemoryType dest_mem_type,
//     rmm::cuda_stream_view stream
// ) {
//     switch (dest_mem_type) {
//     case MemoryType::HOST:
//         std::memcpy(dest, src, size);
//         break;
//     case MemoryType::DEVICE:
//         RAPIDSMP_CUDA_TRY_ALLOC(
//             cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream)
//         );
//         break;
//     default:
//         RAPIDSMP_FAIL("MemoryType: unknown");
//     }
// }

// void copy_buf_to_buf(
//     Buffer&& src_buf, void* dest, MemoryType dest_mem_type, rmm::cuda_stream_view
//     stream
// ) {
//     RAPIDSMP_EXPECTS(
//         src_buf.mem_type == dest_mem_type,
//         "Source and destination buffers must be of the same memory type"
//     );
//     switch (dest_mem_type) {
//     case MemoryType::HOST:
//         std::memcpy(dest, src_buf.data(), src_buf.size);
//         break;
//     case MemoryType::DEVICE:
//         RAPIDSMP_CUDA_TRY_ALLOC(
//             cudaMemcpyAsync(dest, src_buf.data(), src_buf.size, cudaMemcpyDefault,
//             stream)
//         );
//         break;
//     default:
//         RAPIDSMP_FAIL("MemoryType: unknown");
//     }
// }

ChunkBatch ChunkBatch::create(
    Rank dest_rank,
    std::vector<Chunk>&& chunks,
    BufferResource* br,
    rmm::cuda_stream_view stream
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
    header->dest_rank = dest_rank;
    header->num_chunks = chunks.size();

    if (chunks.empty()) {
        return batch;
    }

    size_t metadata_offset = sizeof(BatchHeader); // skip the header
    for (auto&& chunk : chunks) {
        chunk.to_metadata_message(*batch.metadata_buffer_, metadata_offset);
        metadata_offset += (sizeof(Chunk::MetadataMessageHeader) + chunk.metadata_size);
    }


    // std::vector<size_t> metadata_psum{chunks.size(), 0};
    // std::vector<size_t> payload_psum{chunks.size(), 0};
    // // traverse all chunks to determine the partial sum of sizes

    // metadata_psum[0] = chunks[0].metadata->size();
    // payload_psum[0] = chunks[0].gpu_data->size;
    // MemoryType mem_type = chunks[0].gpu_data->mem_type;
    // for (size_t i = 1; i < chunks.size(); ++i) {
    //     metadata_psum[i] = metadata_psum[i - 1] + chunks[i].metadata->size();
    //     payload_psum[i] = payload_psum[i - 1] + chunks[i].gpu_data->size;

    //     RAPIDSMP_EXPECTS(
    //         mem_type == chunks[i].gpu_data->mem_type,
    //         "All chunks in a batch should be of the same memory type"
    //     );
    // }


    // // allocate a buffer to hold the metadata prefix sum, and metadata data buffers
    // batch.metadata_data = std::make_unique<std::vector<uint8_t>>(
    //     sizeof(size_t) * chunks.size() + metadata_psum.back()
    // );
    // // first copy the prefix sum into the buffer
    // size_t medatadata_offset = metadata_psum.size() * sizeof(size_t);
    // std::memcpy(batch.metadata_data->data(), metadata_psum.data(), medatadata_offset);

    // // allocate a buffer to hold the payload prefix sum and data
    // size_t payload_buf_size = sizeof(size_t) * chunks.size() + payload_psum.back();
    // auto [reservation, _] = br->reserve(mem_type, payload_buf_size, false);
    // RAPIDSMP_EXPECTS(reservation.size() == payload_buf_size, "Unable to reserve
    // memory"); batch.payload_data = br->allocate(mem_type, payload_buf_size, stream,
    // reservation); RAPIDSMP_EXPECTS(reservation.size() == 0, "didn't use all of the
    // reservation");

    // // copy the prefix sums to the payload_data buffer
    // // size_t payload_offset = payload_psum.size() * sizeof(size_t);
    // copy_host_to_buf(
    //     payload_psum.data(),
    //     payload_psum.size() * sizeof(size_t),
    //     batch.payload_data->data(),
    //     mem_type,
    //     stream
    // );


    // // traverse the chunks, copy metadata and payloads to the buffers
    // for (auto&& chunk : chunks) {
    //     // copy metadata
    //     std::vector<uint8_t> metadata = *std::move(chunk.metadata);
    //     std::memcpy(
    //         batch.metadata_data->data() + medatadata_offset,
    //         metadata.data(),
    //         metadata.size()
    //     );
    //     medatadata_offset += metadata.size();

    //     // copy payload
    // }


    return batch;
}

bool rapidsmp::shuffler::detail::ChunkBatchReader::has_next() const {
    return current_idx_ < batch_.size();
}

Chunk rapidsmp::shuffler::detail::ChunkBatchReader::next() {
    auto chunk_header = reinterpret_cast<Chunk::MetadataMessageHeader const*>(batch_.)
    // return Chunk();
}

}  // namespace rapidsmp::shuffler::detail