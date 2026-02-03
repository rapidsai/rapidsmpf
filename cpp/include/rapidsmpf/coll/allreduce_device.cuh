/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef __CUDACC__
#error "allreduce_device.cuh must be compiled with NVCC (__CUDACC__ defined)"
#endif

#include <cstddef>
#include <cstdint>

#include <cuda/std/functional>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>

namespace rapidsmpf::coll::detail::device {

/**
 * @brief Kernel function for device-side byte-wise reduction.
 *
 * @param accum The accumulator buffer.
 * @param incoming The incoming buffer.
 * @param element_size Size of each element in bytes.
 * @param count Number of elements to reduce.
 * @param op The device-side byte-wise reduction operator.
 */
template <typename DeviceOp>
__global__ void device_bytewise_reduce_kernel(
    std::byte* accum,
    std::byte const* incoming,
    std::size_t element_size,
    std::size_t count,
    DeviceOp op
) {
    auto const idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        auto* acc_elem = reinterpret_cast<void*>(accum + idx * element_size);
        auto const* in_elem =
            reinterpret_cast<void const*>(incoming + idx * element_size);
        op(acc_elem, in_elem);
    }
}

/**
 * @brief Create a device-based element-wise reduction operator implementation using a
 * byte-wise operator.
 *
 * @param element_size Size of each element in bytes.
 * @param op The device-side byte-wise reduction operator.
 *
 * @return The wrapped reduction operator.
 */
template <typename DeviceOp>
ReduceOperatorFunction make_device_byte_reduce_operator(
    std::size_t element_size, DeviceOp op
) {
    return [element_size,
            op = std::move(op)](PackedData& accum, PackedData&& incoming) mutable {
        RAPIDSMPF_EXPECTS(
            accum.data && incoming.data,
            "Device reduction operator requires non-null buffers"
        );

        auto* acc_buf = accum.data.get();
        auto* in_buf = incoming.data.get();

        auto const acc_nbytes = acc_buf->size;
        auto const in_nbytes = in_buf->size;
        RAPIDSMPF_EXPECTS(
            acc_nbytes == in_nbytes,
            "AllReduce device reduction requires equal-sized buffers"
        );
        RAPIDSMPF_EXPECTS(
            element_size > 0 && acc_nbytes % element_size == 0,
            "AllReduce device reduction buffer size must be multiple of element_size"
        );

        auto const count = acc_nbytes / element_size;
        if (count == 0) {
            return;
        }

        RAPIDSMPF_EXPECTS(
            acc_buf->mem_type() == MemoryType::DEVICE
                && in_buf->mem_type() == MemoryType::DEVICE,
            "Device reduction operator expects device memory"
        );

        cuda_stream_join(acc_buf->stream(), in_buf->stream());

        acc_buf->write_access([&](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
            auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());

            constexpr int threads_per_block = 256;
            auto const blocks = static_cast<std::size_t>(
                (count + threads_per_block - 1) / threads_per_block
            );

            device_bytewise_reduce_kernel<<<
                blocks,
                threads_per_block,
                0,
                stream.value()>>>(acc_bytes, in_bytes, element_size, count, op);
            RAPIDSMPF_CUDA_TRY(cudaGetLastError());
        });
    };
}

}  // namespace rapidsmpf::coll::detail::device
