/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace rapidsmpf {

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
    return cudf::detail::memcpy_async(dst, src, count, stream);
}

}  // namespace rapidsmpf
