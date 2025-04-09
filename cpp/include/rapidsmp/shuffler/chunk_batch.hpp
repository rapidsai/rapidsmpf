/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <vector>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/shuffler/chunk.hpp>

namespace rapidsmp::shuffler::detail {

/**
 * @brief A class representing a batch of chunks.
 *
 * It serializes each chunk data into a metadata buffer and a payload buffer. Additional
 * information such as chunk batch ID, etc are also injected at the front of the metadata
 * buffer.
 *
 * Metadata buffer format:
 * | BatchHeader | [[MetadataMessageHeader, Metadata], ...] |
 *
 * Payload buffer format:
 * | [[Data, ...] |
 *
 */
class ChunkBatch {
    friend class ChunkBatchMetadataReader;

  public:
    /// @brief The size of the chunk metadata header in bytes.
    static constexpr std::ptrdiff_t chunk_metadata_header_size =
        sizeof(Chunk::MetadataMessageHeader);

    /// @brief The structure of the batch header.
    /// @note This is allocated at the front of the the metadata buffer.
    struct BatchHeader {
        uint32_t id;  ///< The id of the batch.
        Rank dest_rank;  ///< The destination rank of the batch.
        size_t num_chunks;  ///< The number of chunks in the batch.
    };

    /// @brief The size of the batch header in bytes.
    static constexpr std::ptrdiff_t batch_header_size = sizeof(BatchHeader);

    /// @brief  Access the BatchHeader of the chunk batch.
    /// @return BatchHeader const* A pointer to the batch header.
    [[nodiscard]] BatchHeader const* header() const {
        RAPIDSMP_EXPECTS(metadata_buffer_, "metadata buffer is null");
        // Maybe converted to constexpr in C++20
        return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data());
    }

    /// @brief Access the destination rank of the chunk batch.
    /// @return Rank The destination rank of the chunk batch.
    [[nodiscard]] Rank destination() const {
        return header()->dest_rank;
    }

    /// @brief Access the number of chunks in the chunk batch.
    /// @return The number of chunks in the chunk batch.
    [[nodiscard]] size_t size() const {
        return header()->num_chunks;
    }

    /// @brief Access the id of the chunk batch.
    /// @return The id of the chunk batch.
    [[nodiscard]] uint32_t id() const {
        return header()->id;
    }

    /**
     * @brief Releases the metadata buffer of the chunk batch.
     *
     * @return The released metadata buffer.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> release_metadata() {
        return std::move(metadata_buffer_);
    }

    /**
     * @brief Releases the payload buffer of the chunk batch.
     *
     * @return The released payload buffer.
     */
    [[nodiscard]] std::unique_ptr<Buffer> release_payload() {
        return std::move(payload_data_);
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
    [[nodiscard]] static ChunkBatch create(
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
    [[nodiscard]] static ChunkBatch create(
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<Buffer> payload_data
    );

    /**
     * @brief Visits the chunk data in the batch.
     * @tparam VisitorFn Visitor function type. Must be callable with the following
     * signature:
     * void(Chunk::MetadataMessageHeader const* chunk_header,
     *      std::vector<uint8_t> const& metadata_buf,
     *      std::ptrdiff_t metadata_offset,
     *      Buffer const& payload_buf,
     *      std::ptrdiff_t payload_offset)
     * @param visitor visitor function
     */
    template <typename VisitorFn>
    void visit_chunk_data(VisitorFn visitor) const {
        assert(metadata_buffer_);
        assert(metadata_buffer_->size() >= batch_header_size);

        std::ptrdiff_t metadata_offset = batch_header_size;
        std::ptrdiff_t payload_offset = 0;

        for (size_t i = 0; i < header()->num_chunks; ++i) {
            assert(
                std::ptrdiff_t(metadata_buffer_->size())
                >= metadata_offset + chunk_metadata_header_size
            );

            auto const* chunk_header =
                reinterpret_cast<Chunk::MetadataMessageHeader const*>(
                    metadata_buffer_->data() + metadata_offset
                );
            metadata_offset += chunk_metadata_header_size;

            assert(
                metadata_buffer_->size()
                >= size_t(metadata_offset) + chunk_header->metadata_size
            );

            assert(payload_data_);
            assert(
                payload_data_->size
                >= size_t(payload_offset) + chunk_header->gpu_data_size
            );

            visitor(
                chunk_header,
                *metadata_buffer_,
                metadata_offset,
                *payload_data_,
                payload_offset
            );

            metadata_offset += std::ptrdiff_t(chunk_header->metadata_size);
            payload_offset += std::ptrdiff_t(chunk_header->gpu_data_size);
        }
    }

    /**
     * @brief Visits the chunks in the batch (by copy).
     * @tparam VisitorFn Visitor function type. Must be callable with the following
     * signature:
     * void(Chunk&& chunk)
     * @param stream The stream to use for copying the chunk data.
     * @param visitor visitor function
     */
    template <typename VisitorFn>
    void visit_chunks(VisitorFn visitor, rmm::cuda_stream_view stream) const {
        visit_chunk_data([&](Chunk::MetadataMessageHeader const* chunk_header,
                             std::vector<uint8_t> const& metadata_buf,
                             std::ptrdiff_t metadata_offset,
                             Buffer const& payload_buf,
                             std::ptrdiff_t payload_offset) {
            std::unique_ptr<std::vector<uint8_t>> chunk_metadata;
            if (chunk_header->metadata_size > 0) {
                chunk_metadata = std::make_unique<std::vector<uint8_t>>(
                    metadata_buf.begin() + metadata_offset,
                    metadata_buf.begin() + metadata_offset
                        + std::ptrdiff_t(chunk_header->metadata_size)
                );
            }

            std::unique_ptr<Buffer> chunk_payload;
            if (chunk_header->gpu_data_size > 0) {
                chunk_payload = payload_buf.copy_slice(
                    payload_offset, std::ptrdiff_t(chunk_header->gpu_data_size), stream
                );
            }

            visitor(Chunk{
                chunk_header->pid,
                chunk_header->cid,
                chunk_header->expected_num_chunks,
                chunk_header->gpu_data_size,
                std::move(chunk_metadata),
                std::move(chunk_payload),
            });
        });
    }

    /**
     * @brief Get all chunks in the batch.
     * @param stream The stream to use for copying the chunk data.
     * @return std::vector<Chunk> A vector of chunks.
     */
    std::vector<Chunk> get_chunks(rmm::cuda_stream_view stream) const {
        std::vector<Chunk> chunks;
        chunks.reserve(header()->num_chunks);
        visit_chunks(
            [&](Chunk&& chunk) { chunks.emplace_back(std::move(chunk)); }, stream
        );
        return chunks;
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
