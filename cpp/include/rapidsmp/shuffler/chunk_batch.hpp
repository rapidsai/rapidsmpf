/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vector>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/shuffler/chunk.hpp>

namespace rapidsmp::shuffler::detail {

class ChunkBatch {
  public:
    struct BatchHeader {
        Rank dest_rank;
        size_t num_chunks;
    };

    Rank destination() const {
        return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data())->dest_rank;
    }

    size_t constexpr size() const {
        return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data())->num_chunks;
    }

    /**
     * @brief Creates a chunk batch.
     *
     * @throws std::logic_error if the memory types of each chunk gpu data is not the same
     */
    static ChunkBatch create(
        Rank dest_rank,
        std::vector<Chunk>&& chunks,
        BufferResource* br,
        rmm::cuda_stream_view stream
    );

  private:
    ChunkBatch() = default;

    // Rank const dest_rank_;  // destination rank
    // size_t const num_chunks_;  // number of chunks in the batch

    /// A buffer containing the BatchHeader, and metadata header and metadata of each
    /// chunk.
    /// |BatchHeader|[[MetadataMessageHeader, Metadata], ...]|
    std::unique_ptr<std::vector<uint8_t>> metadata_buffer_;

    /// GPU data buffer of the packed `cudf::table` associated with this chunk.
    std::unique_ptr<Buffer> payload_data;
};

class ChunkBatchReader {
    ChunkBatchReader(ChunkBatch&& batch);
    bool has_next() const;
    Chunk next();

  private:
    ChunkBatch batch_;
    size_t current_idx_ = 0;
    size_t metadata_offset_ = sizeof(ChunkBatch::BatchHeader);
};

}  // namespace rapidsmp::shuffler::detail