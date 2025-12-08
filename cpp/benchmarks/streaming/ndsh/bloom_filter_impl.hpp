/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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
struct aligned_buffer {
    /**
     * @brief Construct the buffer.
     *
     * @param size The buffer size.
     * @param alignment The requested alignment.
     * @param stream Stream for allocations.
     * @param mr Memory resource for allocations.
     */
    explicit aligned_buffer(
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
    ~aligned_buffer() {
        mr.deallocate(stream, data, size, alignment);
    }

    aligned_buffer(aligned_buffer const&) = delete;
    aligned_buffer& operator=(aligned_buffer const&) = delete;

    aligned_buffer(aligned_buffer&& other)
        : size{other.size},
          alignment{other.alignment},
          stream{other.stream},
          mr{other.mr},
          data{std::exchange(other.data, nullptr)} {}

    aligned_buffer& operator=(aligned_buffer&& other) {
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

    std::size_t size;
    std::size_t alignment;
    rmm::cuda_stream_view stream;
    rmm::device_async_resource_ref mr;
    void* data;
};

/**
 * @brief Create device storage for the bloom filter.
 *
 * @param num_blocks Number of blocks.
 * @param stream CUDA stream for device launches and allocations.
 * @param mr Memory resource for allocations.
 */
aligned_buffer create_filter_storage(
    std::size_t num_blocks,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);

/**
 * @brief Update the filter with fingerprints from a table.
 *
 * @param storage Allocated device storage for the bloom filter
 * @param num_blocks Number of blocks.
 * @param values_to_hash Table of values to hash.
 * @param seed Hash seed
 * @param stream CUDA stream for device launches and allocations.
 * @param mr Memory resource for allocations.
 */
void update_filter(
    aligned_buffer& storage,
    std::size_t num_blocks,
    cudf::table_view const& values_to_hash,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);

/**
 * @brief Apply the filter to fingerprints from a table.
 *
 * @param storage Allocated device storage for the bloom filter
 * @param num_blocks Number of blocks.
 * @param values_to_hash Table of values to hash.
 * @param seed Hash seed
 * @param stream CUDA stream for device launches and allocations.
 * @param mr Memory resource for allocations.
 *
 * @return Mask vector select rows in the table that were selected by the filter.
 */
rmm::device_uvector<bool> apply_filter(
    aligned_buffer& storage,
    std::size_t num_blocks,
    cudf::table_view const& values_to_hash,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);
}  // namespace rapidsmpf::ndsh
