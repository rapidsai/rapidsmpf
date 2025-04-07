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
    friend class ChunkBatchMetadataReader;

  public:
    struct BatchHeader {
        uint32_t id;
        Rank dest_rank;
        size_t num_chunks;
    };

    Rank destination() const;
    
    size_t size() const;

    uint32_t id() const;

    /**
     * @brief Creates a chunk batch.
     *
     * @throws std::logic_error if the memory types of each chunk gpu data is not the same
     */
    static ChunkBatch create(
      uint32_t id,
        Rank dest_rank,
        std::vector<Chunk>&& chunks,
        BufferResource* br,
        rmm::cuda_stream_view stream
    );

    static ChunkBatch create(
      std::unique_ptr<std::vector<uint8_t>> metadata,
      std::unique_ptr<Buffer> payload_data
    );

  private:
    ChunkBatch() = default;

    ChunkBatch(
      std::unique_ptr<std::vector<uint8_t>> metadata,
      std::unique_ptr<Buffer> payload_data
    );

    /// A buffer containing the BatchHeader, and metadata header and metadata of each
    /// chunk.
    /// |BatchHeader|[[MetadataMessageHeader, Metadata], ...]|
    std::unique_ptr<std::vector<uint8_t>> metadata_buffer_;

    /// GPU data buffer of the packed `cudf::table` associated with this chunk.
    std::unique_ptr<Buffer> payload_data_;
};

class ChunkBatchMetadataReader {
    ChunkBatchMetadataReader(ChunkBatch const& batch);
    bool has_next() const;
    Chunk next();

  private:
    const ChunkBatch& batch_;
    size_t current_idx_ = 0;
    size_t metadata_offset_ = sizeof(ChunkBatch::BatchHeader);
};

}  // namespace rapidsmp::shuffler::detail