/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>

namespace rapidsmpf {

/**
 * @brief Asynchronously copies a batch of buffers using the most efficient available API.
 *
 * On CUDA 13.0+ with a non-default stream, uses `cudaMemcpyBatchAsync` with
 * `cudaMemcpySrcAccessOrderStream`, which defers reading the source buffers until
 * the stream reaches each copy. This enables true asynchronous copies from pageable
 * host memory on modern systems with HMM/ATS support.
 *
 * Falls back to per-copy `cudaMemcpyAsync` on older CUDA versions or when the default
 * stream is used.
 *
 * @param dsts Host pointer to a list of destination pointers.
 * @param srcs Host pointer to a list of source pointers.
 * @param sizes Host pointer to a list of sizes (bytes).
 * @param count Number of entries in dsts, srcs, sizes.
 * @param stream CUDA stream on which copies are enqueued.
 * @return cudaError_t CUDA error code.
 */
[[nodiscard]] inline cudaError_t cuda_memcpy_batch_async(
    void* const* dsts,
    void const* const* srcs,
    std::size_t const* sizes,
    std::size_t count,
    rmm::cuda_stream_view stream
) {
#if CUDART_VERSION >= 13000
    if (!stream.is_default()) {
        // Filter out invalid copies; cudaMemcpyBatchAsync does not support
        // nullptr dst/src or size==0.
        auto is_invalid = [&](std::size_t i) {
            return dsts[i] == nullptr || srcs[i] == nullptr || sizes[i] == 0;
        };

        std::vector<void*> valid_dsts;
        std::vector<void const*> valid_srcs;
        std::vector<std::size_t> valid_sizes;

        bool has_invalid = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (is_invalid(i)) {
                has_invalid = true;
                break;
            }
        }

        if (has_invalid) {
            valid_dsts.reserve(count);
            valid_srcs.reserve(count);
            valid_sizes.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                if (dsts[i] != nullptr && srcs[i] != nullptr && sizes[i] != 0) {
                    valid_dsts.push_back(dsts[i]);
                    valid_srcs.push_back(srcs[i]);
                    valid_sizes.push_back(sizes[i]);
                }
            }
            if (valid_dsts.empty()) {
                return cudaSuccess;
            }
            dsts = valid_dsts.data();
            srcs = valid_srcs.data();
            sizes = valid_sizes.data();
            count = valid_dsts.size();
        }

        cudaMemcpyAttributes attrs = {
            .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
            .flags = cudaMemcpyFlagPreferOverlapWithCompute
        };
        std::size_t attrs_idxs = 0;
        return cudaMemcpyBatchAsync(
            dsts, srcs, sizes, count, &attrs, &attrs_idxs, 1, stream.value()
        );
    }
#endif  // CUDART_VERSION >= 13000
    for (std::size_t i = 0; i < count; ++i) {
        if (dsts[i] == nullptr || srcs[i] == nullptr || sizes[i] == 0) {
            continue;
        }
        cudaError_t status = cudaMemcpyAsync(
            dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value()
        );
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

/**
 * @brief Asynchronously copies memory between host and/or device buffers.
 *
 * The copy direction is inferred from the pointer types (`cudaMemcpyDefault`).
 * The source buffer must remain valid until the stream executes the copy.
 *
 * This function should be used instead of `cudaMemcpyAsync`, as it provides
 * improved semantics for asynchronous copies, especially from pageable host memory.
 *
 * ## Background
 *
 * The legacy `cudaMemcpyAsync` API accesses non-CUDA-registered host pointers
 * (e.g., allocations from `malloc` or `new`) at the time of the API call,
 * rather than in stream order. This behavior originates from earlier GPU
 * architectures that could not directly access such memory, requiring an
 * immediate CPU-side staging step.
 *
 * Modern systems with HMM/ATS allow GPUs to access these pointers directly.
 * However, the semantics of `cudaMemcpyAsync` cannot be changed without
 * breaking existing code. The batched memcpy APIs (e.g.,
 * `cudaMemcpyBatchAsync`) introduced in CUDA 13.0 allow the caller to specify
 * `cudaMemcpySrcAccessOrderStream`, ensuring that the source is accessed in
 * stream order and enabling true asynchronous copies from pageable host memory.
 *
 * @param dst Destination memory address.
 * @param src Source memory address.
 * @param count Number of bytes to copy.
 * @param stream CUDA stream on which the copy is enqueued.
 * @return cudaError_t CUDA error code.
 */
[[nodiscard]] inline cudaError_t cuda_memcpy_async(
    void* dst, void const* src, std::size_t count, rmm::cuda_stream_view stream
) {
    if (count == 0) {
        return cudaSuccess;
    }
    void const* src_ptr = src;
    return cuda_memcpy_batch_async(&dst, &src_ptr, &count, 1, stream);
}

}  // namespace rapidsmpf
