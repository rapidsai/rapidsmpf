/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <rmm/cuda_stream_view.hpp>

namespace rapidsmpf {

enum class Algo {
    Cascaded,
    LZ4,
    Zstd,
    Snappy
};

/**
 * @brief Parameters for nvCOMP codec configuration
 *
 * Holds configuration parameters for both generic and algorithm-specific compression
 * settings.
 */
struct KvParams {
    /// Chunk size for compression operations (default: 1 MiB)
    std::size_t chunk_size{1 << 20};

    /// Number of run-length encoding passes in Cascaded codec (must be non-negative,
    /// default: 1)
    int cascaded_rle{1};

    /// Number of delta encoding passes in Cascaded codec (must be non-negative, default:
    /// 1)
    int cascaded_delta{1};

    /// Enable bitpacking in Cascaded codec (default: enabled)
    bool cascaded_bitpack{true};
};

/**
 * @brief Abstract base class for nvCOMP codec implementations
 *
 * Provides a unified interface for different compression algorithms (LZ4, Cascaded, etc.)
 * to perform compression and decompression operations on GPU device memory.
 */
class NvcompCodec {
  public:
    virtual ~NvcompCodec() = default;

    /**
     * @brief Calculate the maximum compressed size for the given input size
     *
     * @param uncompressed_bytes Size of the uncompressed data in bytes
     * @param stream CUDA stream for operations
     * @return Maximum possible compressed size in bytes
     */
    virtual std::size_t get_max_compressed_bytes(
        std::size_t uncompressed_bytes, rmm::cuda_stream_view stream
    ) = 0;

    /**
     * @brief Compress data on the GPU
     *
     * @param d_in Pointer to uncompressed data on device
     * @param in_bytes Size of uncompressed data in bytes
     * @param d_out Pointer to output buffer on device for compressed data
     * @param out_bytes Pointer to output variable that will be set to actual compressed
     * size
     * @param stream CUDA stream for operations
     */
    virtual void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream
    ) = 0;

    /**
     * @brief Decompress data on the GPU
     *
     * @param d_in Pointer to compressed data on device
     * @param in_bytes Size of compressed data in bytes
     * @param d_out Pointer to output buffer on device for decompressed data
     * @param out_bytes Expected size of decompressed data in bytes
     * @param stream CUDA stream for operations
     */
    virtual void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) = 0;
};

/**
 * @brief Create an nvCOMP codec instance
 *
 * @param algo The compression algorithm to use
 * @param p Parameters for the codec
 * @return A unique pointer to an NvcompCodec instance
 */
std::unique_ptr<NvcompCodec> make_codec(Algo algo, KvParams const& p);

}  // namespace rapidsmpf
