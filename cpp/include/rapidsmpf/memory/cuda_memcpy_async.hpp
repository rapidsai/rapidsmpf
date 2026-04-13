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
 * The source buffer must remain valid until the stream has executed the copy.
 *
 * @note Currently delegates to `cudf::detail::memcpy_async`, which uses
 * `cudaMemcpyBatchAsync` on CUDA 13.0+ with `cudaMemcpySrcAccessOrderStream`.
 * The underlying implementation may change in the future; all callers should
 * use this function so that any such change is applied uniformly.
 *
 * @param dst    Destination memory address.
 * @param src    Source memory address.
 * @param count  Number of bytes to copy.
 * @param stream CUDA stream on which the copy is enqueued.
 * @return cudaError_t CUDA error code.
 */
[[nodiscard]] inline cudaError_t cuda_memcpy_async(
    void* dst, void const* src, std::size_t count, rmm::cuda_stream_view stream
) {
    return cudf::detail::memcpy_async(dst, src, count, stream);
}

}  // namespace rapidsmpf
