/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

class ChunkForwardIterator;

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
    friend class ChunkForwardIterator;

  public:
    using iterator = ChunkForwardIterator;  ///< Chunk iterator type

    /**
     * @brief The size of the chunk metadata header in bytes.
     */
    static constexpr std::ptrdiff_t chunk_metadata_header_size =
        sizeof(Chunk::MetadataMessageHeader);

    /**
     * @brief The structure of the batch header.
     * @note This is allocated at the front of the the metadata buffer.
     */
    struct BatchHeader {
        uint32_t id;  ///< The id of the batch.
        Rank dest_rank;  ///< The destination rank of the batch.
        size_t num_chunks;  ///< The number of chunks in the batch.
    };

    /** @brief The size of the batch header in bytes. */
    static constexpr std::ptrdiff_t batch_header_size = sizeof(BatchHeader);

    /**
     * @brief Access the BatchHeader of the chunk batch.
     * @return A pointer to the batch header.
     */
    [[nodiscard]] BatchHeader const* header() const {
        RAPIDSMPF_EXPECTS(metadata_buffer_, "metadata buffer is null");
        // Maybe converted to constexpr in C++20
        return reinterpret_cast<BatchHeader const*>(metadata_buffer_->data());
    }

    /**
     * @brief Access the destination rank of the chunk batch.
     * @return The destination rank of the chunk batch.
     */
    [[nodiscard]] Rank destination() const {
        return header()->dest_rank;
    }

    /**
     * @brief Access the number of chunks in the chunk batch.
     * @return The number of chunks in the chunk batch.
     */
    [[nodiscard]] size_t size() const {
        return header()->num_chunks;
    }

    /**
     * @brief Access the id of the chunk batch.
     * @return The id of the chunk batch.
     */
    [[nodiscard]] uint32_t id() const {
        return header()->id;
    }

    /**
     * @brief Releases the ownership of the metadata buffer of the chunk batch.
     *
     * @return The released metadata buffer.
     *
     * @throws std::logic_error if the metadata buffer is null.
     *
     * @note This method should be called only once.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> release_metadata() {
        RAPIDSMPF_EXPECTS(metadata_buffer_, "metadata buffer is null");
        return std::move(metadata_buffer_);
    }

    /**
     * @brief Releases the ownership of the payload buffer of the chunk batch.
     *
     * @return The released payload buffer. If there was no payload, returns nullptr.
     *
     * @note Unlike metadata buffer, payload buffer could be null.
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
     * @return New chunk batch.
     *
     * @throws std::logic_error if the payload buffer memory types of the chunks are not
     * the same. Chunk data is expected to be of the same memory type.
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
     * @return New chunk batch.
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

            if (chunk_header->gpu_data_size > 0) {
                assert(payload_data_);
                assert(
                    payload_data_->size
                    >= size_t(payload_offset) + chunk_header->gpu_data_size
                );
            }

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
     * @brief Iterator to the beginning of the Chunk batch
     * @param stream The stream to use for any memory allocations.
     * @return Forward iterator to the beginning
     */
    iterator begin(rmm::cuda_stream_view stream) const;

    /**
     * @brief Iterator to the end of the Chunk batch
     * @param stream The stream to use for any memory allocations.
     * @return Forward iterator to the end
     */
    iterator end(rmm::cuda_stream_view stream) const;

  private:
    ChunkBatch(
        std::unique_ptr<std::vector<uint8_t>> metadata,
        std::unique_ptr<Buffer> payload_data
    );

    /// A buffer containing the BatchHeader, and metadata header and metadata of each
    /// chunk. This buffer will not be null.
    /// |BatchHeader|[[MetadataMessageHeader, Metadata], ...]|
    ///
    /// TODO: change the format to have thhe MetadataMessageHeaders at the front (after
    /// BatchHeader), followed by the metadata. This will be a more cache efficient
    /// traversal pattern.
    std::unique_ptr<std::vector<uint8_t>> metadata_buffer_;

    /// GPU data buffer of the packed `cudf::table` associated with this chunk. This
    /// buffer could be null if there were no chunks with payload.
    std::unique_ptr<Buffer> payload_data_;

    /// cached begin iterator, as creating an iterator object is relatively
    /// expensive. This allows calling `begin()` trivially multiple times.
    mutable std::unique_ptr<iterator> cached_begin_iter_;
};

/**
 * @brief Forward iterator of chunks in the chunk batch.
 */
class ChunkForwardIterator {
    friend ChunkBatch;

  public:
    using iterator_category = std::forward_iterator_tag;  ///< iterator category def
    using value_type = Chunk;  ///< value type def
    using difference_type = std::ptrdiff_t;  ///< difference type def
    using pointer = Chunk*;  ///< pointer type def
    using reference = Chunk&;  ///< rederence type def

    /**
     * @brief Return the reference to the chunk in the iterator.
     * @return chunk ref
     */
    reference operator*() const {
        return *chunk_;
    }

    /**
     * @brief Return the pointer to the chunk in the iterator.
     * @return chunk ptr
     */
    pointer operator->() const {
        return chunk_.get();
    }

    /**
     * @brief Prefix increment of the iterator.
     * @return Reference to the incremented iterator
     */
    ChunkForwardIterator& operator++();

    /**
     * @brief Postfix increment of the iterator.
     * @return Copy of the iterator before increment
     */
    ChunkForwardIterator operator++(int);

    /**
     * @brief Equality comparison of iterators.
     * @param other The other iterator to compare with
     * @return true if the iterators are equal, false otherwise
     */
    bool operator==(ChunkForwardIterator const& other) const {
        // Only check the metadata buffer, as it has sufficient information to check
        // equality
        if (&batch_ == &other.batch_ && metadata_offset_ == other.metadata_offset_) {
            // if equal, make sure pointer offsets also match.
            assert(payload_offset_ == other.payload_offset_);
            return true;
        }

        return false;
    }

    /**
     * @brief Inequality comparison of iterators.
     * @param other The other iterator to compare with
     * @return true if the iterators are not equal, false otherwise
     */
    bool operator!=(ChunkForwardIterator const& other) const;

    /**
     * @brief Copy constructor.
     * @param other The other iterator to copy from.
     */
    ChunkForwardIterator(const ChunkForwardIterator& other) = default;

    /**
     * @brief Copy assignment operator.
     * @param other The other iterator to copy from.
     * @return Reference to the assigned iterator.
     */
    ChunkForwardIterator& operator=(const ChunkForwardIterator& other) = default;

  private:
    /**
     * @brief Private constructor. Use ChunkBatch to access the begin and end iterators.
     *
     * @param batch The chunk batch.
     * @param metadata_offset The current metadata offset to the chunk boundary.
     * @param payload_offset The current payload offset to the chunk gpu data.
     * @param stream The stream to use for any memory allocations.
     */
    ChunkForwardIterator(
        ChunkBatch const& batch,
        std::ptrdiff_t metadata_offset,
        std::ptrdiff_t payload_offset,
        rmm::cuda_stream_view stream
    );


    /**
     * @brief Advance class members to the next chunk
     */
    void advance_chunk();

    /**
     * @brief Unwrap the current chunk header.
     * @return Chunk header ptr.
     */
    inline Chunk::MetadataMessageHeader const* chunk_header() const {
        return reinterpret_cast<Chunk::MetadataMessageHeader const*>(
            batch_.metadata_buffer_->data() + metadata_offset_
        );
    }

    /**
     * @brief Check if current position contains a chunks.
     * @return True, if metadata offset points to a valid chunk.
     */
    inline bool has_chunk() const {
        return batch_.size() > 0
               && (size_t(metadata_offset_) < batch_.metadata_buffer_->size());
    }

    /**
     * @brief Make a chunk.
     * @return Chunk wrapped in a shared ptr
     */
    static std::shared_ptr<Chunk> make_chunk(
        ChunkBatch const& batch,
        std::ptrdiff_t metadata_offset,
        std::ptrdiff_t payload_offset,
        rmm::cuda_stream_view stream
    );
    ChunkBatch const& batch_;  ///< Reference to the chunk batch being iterated
    std::ptrdiff_t
        metadata_offset_;  ///< metadata offset to the chunk boundary. It points to the
                           ///< beginning of the chunk metadata header.
    std::ptrdiff_t payload_offset_;  ///< payload offset to the chunk gpu data.
    rmm::cuda_stream_view stream_;  ///< Stream for memory operations

    /// Chunk is held on a shared_ptr, so that the iterator object can be trivially
    /// copied. If there were no chunks in the batch, this will be null.
    std::shared_ptr<Chunk> chunk_;  ///< Current chunk being pointed to
};

}  // namespace rapidsmpf::shuffler::detail
