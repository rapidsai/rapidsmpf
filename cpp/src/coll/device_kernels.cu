/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <thrust/functional.h>

#include <cuda/std/functional>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>

namespace rapidsmpf::coll::detail {

using ReduceOperatorFunction = rapidsmpf::coll::ReduceOperatorFunction;

namespace {

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

template <typename DeviceOp>
void device_byte_reduce(
    Buffer* acc_buf, Buffer* in_buf, std::size_t element_size, DeviceOp op
) {
    RAPIDSMPF_EXPECTS(
        acc_buf && in_buf, "Device reduction operator requires non-null buffers"
    );
    RAPIDSMPF_EXPECTS(
        acc_buf->mem_type() == MemoryType::DEVICE
            && in_buf->mem_type() == MemoryType::DEVICE,
        "Device reduction operator expects device memory"
    );

    auto const acc_nbytes = acc_buf->size;
    auto const in_nbytes = in_buf->size;
    RAPIDSMPF_EXPECTS(
        acc_nbytes == in_nbytes, "AllReduce device reduction requires equal-sized buffers"
    );
    RAPIDSMPF_EXPECTS(
        element_size > 0 && acc_nbytes % element_size == 0,
        "AllReduce device reduction buffer size must be multiple of element_size"
    );

    auto const count = acc_nbytes / element_size;

    if (count == 0) {
        return;
    }

    cuda_stream_join(acc_buf->stream(), in_buf->stream());

    acc_buf->write_access([&](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
        auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());

        constexpr int threads_per_block = 256;
        auto const blocks =
            static_cast<std::size_t>((count + threads_per_block - 1) / threads_per_block);

        device_bytewise_reduce_kernel<<<blocks, threads_per_block, 0, stream.value()>>>(
            acc_bytes, in_bytes, element_size, count, op
        );
        RAPIDSMPF_CUDA_TRY(cudaGetLastError());
    });
}

}  // namespace

template <typename DeviceOp>
ReduceOperatorFunction make_device_byte_reduce_operator_impl(
    std::size_t element_size, DeviceOp op
) {
    RAPIDSMPF_EXPECTS(
        element_size > 0, "Device reduction operator requires element_size>0"
    );
    return [element_size,
            op = std::move(op)](PackedData& accum, PackedData&& incoming) mutable {
        device_byte_reduce(accum.data.get(), incoming.data.get(), element_size, op);
    };
}

#define RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(T, OP)                           \
    template ReduceOperatorFunction make_device_byte_reduce_operator_impl( \
        std::size_t element_size, DeviceElementwiseOp<T, OP> op            \
    );

RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(bool, cuda::std::logical_or<bool>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(bool, cuda::std::multiplies<bool>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(int, cuda::std::plus<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(int, cuda::std::multiplies<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(int, cuda::minimum<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(int, cuda::maximum<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(float, cuda::std::plus<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(float, cuda::std::multiplies<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(float, cuda::minimum<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(float, cuda::maximum<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(double, cuda::std::plus<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(double, cuda::std::multiplies<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(double, cuda::minimum<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(double, cuda::maximum<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(std::uint64_t, cuda::std::plus<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(std::uint64_t, cuda::std::multiplies<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(std::uint64_t, cuda::minimum<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(std::uint64_t, cuda::maximum<std::uint64_t>)

#undef RAPIDSMPF_INSTANTIATE_DEVICE_BYTE

#define RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(T, OP) \
    template ReduceOperator make_device_elementwise_reduce_operator<T, OP>(OP op);

RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(bool, cuda::std::logical_or<bool>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(bool, cuda::std::multiplies<bool>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(int, cuda::std::plus<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(int, cuda::std::multiplies<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(int, cuda::minimum<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(int, cuda::maximum<int>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(float, cuda::std::plus<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(float, cuda::std::multiplies<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(float, cuda::minimum<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(float, cuda::maximum<float>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(double, cuda::std::plus<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(double, cuda::std::multiplies<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(double, cuda::minimum<double>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(double, cuda::maximum<double>)
// uint64_t covers unsigned long on most platforms; instantiate only once to avoid
// duplicates.
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::std::plus<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(
    std::uint64_t, cuda::std::multiplies<std::uint64_t>
)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::minimum<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::maximum<std::uint64_t>)

#undef RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE

}  // namespace rapidsmpf::coll::detail
