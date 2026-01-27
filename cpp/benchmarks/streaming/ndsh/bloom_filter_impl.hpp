/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <stdexcept>

#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::ndsh {
/**
 * @brief A type-erased buffer with an allocation with specified alignment.
 */
struct AlignedBuffer {
    /**
     * @brief Construct the buffer.
     *
     * @param size The buffer size.
     * @param alignment The requested alignment.
     * @param stream Stream for allocations.
     * @param mr Memory resource for allocations.
     */
    explicit AlignedBuffer(
        std::size_t size,
        std::size_t alignment,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    )
        : size{size},
          alignment{alignment},
          stream{stream},
          mr{mr},
          data{mr.allocate(stream, size, alignment)} {}

    /**
     * @brief Deallocate the buffer.
     */
    ~AlignedBuffer() {
        mr.deallocate(stream, data, size, alignment);
    }

    AlignedBuffer(AlignedBuffer const&) = delete;
    AlignedBuffer& operator=(AlignedBuffer const&) = delete;

    AlignedBuffer(AlignedBuffer&& other)
        : size{other.size},
          alignment{other.alignment},
          stream{other.stream},
          mr{other.mr},
          data{std::exchange(other.data, nullptr)} {}

    AlignedBuffer& operator=(AlignedBuffer&& other) {
        if (this != &other) {
            RAPIDSMPF_EXPECTS(
                !data,
                "cannot move into an already initialized aligned_buffer",
                std::invalid_argument
            );
        }
        size = other.size;
        alignment = other.alignment;
        stream = other.stream;
        mr = other.mr;
        data = std::exchange(other.data, nullptr);
        return *this;
    }

    std::size_t size;  ///< Size in bytes
    std::size_t alignment;  ///< Alignment in bytes
    rmm::cuda_stream_view stream;  ///< Stream we were allocated on
    rmm::device_async_resource_ref mr;  ///< Memory resource for deallocation
    void* data;  ///< Data
};

/**
 * @brief A bloom filter, used for approximate set membership queries.
 */
struct BloomFilter {
    /**
     * @brief Create a filter.
     *
     * @param num_blocks Number of blocks in the filter.
     * @param seed Seed used for hashing each value.
     * @param stream CUDA stream for allocations and device operations.
     * @param mr Memory resource for allocations.
     */
    BloomFilter(
        std::size_t num_blocks,
        std::uint64_t seed,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    );

    /**
     * @brief Add values to the filter.
     *
     * @param values_to_hash table of values to hash (with cudf::hashing::xxhash_64())
     * @param stream CUDA stream for allocations and device operations.
     * @param mr Memory resource for allocations.
     */
    void add(
        cudf::table_view const& values_to_hash,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    );

    /**
     * @brief Merge two filters, computing their union.
     *
     * @param other Other filter to merge into this one.
     * @param stream CUDA stream for device operations.
     *
     * @throws std::logic_error If `other` is not compatible with this filter.
     */
    void merge(BloomFilter const& other, rmm::cuda_stream_view stream);

    /**
     * @brief Return a mask of which rows are contained in the filter.
     *
     * @param values Value to check for set membership
     * @param stream CUDA stream for allocations and device operations.
     * @param mr Memory resource for allocations.
     */
    [[nodiscard]] rmm::device_uvector<bool> contains(
        cudf::table_view const& values,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    );

    /**
     * @brief @return The stream the underlying storage is valid on.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;

    /**
     * @brief @return Pointer to the underlying storage.
     */
    [[nodiscard]] void* data() const noexcept;

    /**
     * @brief @return Size in bytes of the underlying storage.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief @return Number of blocks to use if the filter should fit in a given L2 cache
     * size.
     *
     * @param l2size Size of the L2 cache in bytes.
     */
    [[nodiscard]] static std::size_t fitting_num_blocks(std::size_t l2size);

  private:
    std::size_t num_blocks_;  ///< Number of blocks used in the filter.
    std::uint64_t seed_;  ///< Seed used when hashing values.
    AlignedBuffer storage_;  ///< Backing storage.
};

}  // namespace rapidsmpf::ndsh
