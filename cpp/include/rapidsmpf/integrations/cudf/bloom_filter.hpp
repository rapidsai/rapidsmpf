/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>

#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

namespace rapidsmpf {

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
    void merge(BloomFilter& other, rmm::cuda_stream_view stream);

    /**
     * @brief Return a mask of which rows are contained in the filter.
     *
     * @param values Value to check for set membership
     * @param stream CUDA stream for allocations and device operations.
     * @param mr Memory resource for allocations.
     *
     * @return Mask vector to be used for filtering the table.
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
    [[nodiscard]] void* data() noexcept;

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
    [[nodiscard]] static std::size_t fitting_num_blocks(std::size_t l2size) noexcept;

  private:
    std::size_t num_blocks_;  ///< Number of blocks used in the filter.
    std::uint64_t seed_;  ///< Seed used when hashing values.
    rmm::device_buffer storage_;  ///< Backing storage.
};

}  // namespace rapidsmpf
