/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/functional.h>

#include <cuda/std/functional>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/coll/allreduce_device.cuh>

namespace rapidsmpf::coll::detail {

#define RAPIDSMPF_INSTANTIATE_DEVICE_BYTE(T, OP)                              \
    template ReduceOperatorFunction device::make_device_byte_reduce_operator( \
        std::size_t element_size, DeviceElementwiseOp<T, OP> op               \
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
    template ReduceOperator make_device_reduce_operator<T, OP>(OP op);

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
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::std::plus<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(
    std::uint64_t, cuda::std::multiplies<std::uint64_t>
)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::minimum<std::uint64_t>)
RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE(std::uint64_t, cuda::maximum<std::uint64_t>)

#undef RAPIDSMPF_INSTANTIATE_DEVICE_ELEMENTWISE

}  // namespace rapidsmpf::coll::detail
