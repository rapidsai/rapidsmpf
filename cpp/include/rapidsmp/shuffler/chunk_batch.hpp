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

/**
 * @brief A class representing a batch of chunks.
 */
class ChunkBatch {
    friend class ChunkBatchMetadataReader;

  public:
    /// @brief The size of the chunk metadata header in bytes.
    static constexpr size_t chunk_metadata_header_size =
        sizeof(Chunk::MetadataMessageHeader);

    /// @brief The structure of the batch header.
    /// @note This is allocated at the front of the the metadata buffer.
    struct BatchHeader {
        uint32_t id;  ///< The id of the batch.
        Rank dest_rank;  ///< The destination rank of the batch.
        size_t num_chunks;  ///< The number of chunks in the batch.
    };

    /// @brief The size of the batch header in bytes.
    static constexpr size_t batch_header_size = sizeof(BatchHeader);

    /// @brief  Access the BatchHeader of the chunk batch.
    /// @return BatchHeader const* A pointer to the batch header.
    BatchHeader const* header() const {
        // Maybe converted to constexpr in C++20
        return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data());
    }

    /// @brief Access the destination rank of the chunk batch.
    /// @return Rank The destination rank of the chunk batch.
    Rank destination() const {
        return header()->dest_rank;
    }

    /// @brief Access the number of chunks in the chunk batch.
    /// @return The number of chunks in the chunk batch.
    size_t size() const {
        return header()->num_chunks;
    }

    /// @brief Access the id of the chunk batch.
    /// @return The id of the chunk batch.
    uint32_t id() const {
        return header()->id;
    }

    /**
     * @brief Creates a chunk batch.
     *
     * @param id The id of the batch.
     * @param dest_rank The destination rank of the batch.
     * @param chunks The chunks to be included in the batch.
     * @param br The buffer resource to use for allocating the metadata buffer.
     * @param stream The stream to use for allocating the metadata buffer.
     * @return ChunkBatch The created chunk batch.
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

    /**
     * @brief Creates a chunk batch from a metadata buffer and a payload buffer.
     *
     * @param metadata The metadata buffer.
     * @param payload_data The payload buffer.
     * @return ChunkBatch The created chunk batch.
     *
     * @throws std::logic_error if the provided buffers violate the format of a chunk
     * batch.
     */
    static ChunkBatch create(
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<Buffer> payload_data
    );

    /**
     * @brief Visits the chunk data in the batch.
     * @tparam VisitorFn Visitor function type. Must be callable with the following
     * signature:
     * void(Chunk::MetadataMessageHeader const* chunk_header,
     *      std::vector<uint8_t> const& metadata_buf,
     *      size_t metadata_offset,
     *      Buffer const& payload_buf,
     *      size_t payload_offset)
     * @param visitor visitor function
     */
    template <typename VisitorFn>
    void visit_chunk_data(VisitorFn visitor) const {
        assert(metadata_buffer_);
        assert(metadata_buffer_->size() >= batch_header_size);

        size_t metadata_offset = batch_header_size;
        size_t payload_offset = 0;

        for (size_t i = 0; i < header()->num_chunks; ++i) {
            assert(
                metadata_buffer_->size() >= metadata_offset + chunk_metadata_header_size
            );

            auto const* chunk_header =
                reinterpret_cast<Chunk::MetadataMessageHeader const*>(
                    metadata_buffer_->data() + metadata_offset
                );
            metadata_offset += chunk_metadata_header_size;

            assert(
                metadata_buffer_->size() >= metadata_offset + chunk_header->metadata_size
            );

            assert(payload_data_);
            assert(payload_data_->size >= payload_offset + chunk_header->payload_size);

            visitor(
                chunk_header,
                *metadata_buffer_,
                metadata_offset,
                *payload_data_,
                payload_offset
            );

            metadata_offset += chunk_header->metadata_size;
            payload_offset += chunk_header->gpu_data_size;
        }
    }

  private:
    ChunkBatch(
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<Buffer> payload_data
    );

    /// A buffer containing the BatchHeader, and metadata header and metadata of each
    /// chunk.
    /// |BatchHeader|[[MetadataMessageHeader, Metadata], ...]|
    ///
    /// TODO: change the format to have thhe MetadataMessageHeaders at the front (after
    /// BatchHeader), followed by the metadata. This will be a more cache efficient
    /// traversal pattern.
    std::unique_ptr<std::vector<uint8_t>> metadata_buffer_;

    /// GPU data buffer of the packed `cudf::table` associated with this chunk.
    std::unique_ptr<Buffer> payload_data_;
};

// class ChunkBatchMetadataReader {
//     /// @brief
//     /// @param batch
//     /// @param stream
//     ChunkBatchMetadataReader(ChunkBatch const& batch, rmm::cuda_stream_view stream);

//     /// @brief
//     /// @return
//     bool has_next() const;

//     /// @brief
//     /// @return
//     Chunk next();

//   private:
//     const ChunkBatch& batch_;
//     rmm::cuda_stream_view stream_;
//     size_t metadata_offset_ = sizeof(ChunkBatch::BatchHeader);
//     size_t payload_offset_ = 0;
// };

}  // namespace rapidsmp::shuffler::detail
