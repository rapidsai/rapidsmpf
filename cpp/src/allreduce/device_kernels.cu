/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <rmm/exec_policy.hpp>

#include <rapidsmpf/allreduce/allreduce.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>

namespace rapidsmpf::allreduce::detail {

namespace {

template <typename T, typename Op>
void device_elementwise_reduce(Buffer* acc_buf, Buffer* in_buf, Op op) {
    RAPIDSMPF_EXPECTS(
        acc_buf && in_buf, "Device reduction kernel requires non-null buffers"
    );
    RAPIDSMPF_EXPECTS(
        acc_buf->mem_type() == MemoryType::DEVICE
            && in_buf->mem_type() == MemoryType::DEVICE,
        "Device reduction kernel expects device memory"
    );

    auto const acc_nbytes = acc_buf->size;
    auto const in_nbytes = in_buf->size;
    RAPIDSMPF_EXPECTS(
        acc_nbytes == in_nbytes, "AllReduce device reduction requires equal-sized buffers"
    );
    RAPIDSMPF_EXPECTS(
        acc_nbytes % sizeof(T) == 0,
        "AllReduce device reduction buffer size must be multiple of sizeof(T)"
    );

    auto const count = acc_nbytes / sizeof(T);

    // Input buffer: assume ready as provided by the communicator / AllReduce.
    auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());
    auto const* in_ptr = reinterpret_cast<T const*>(in_bytes);

    // Destination buffer: use write_access so the latest-write event is updated.
    acc_buf->write_access([&](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
        auto* acc_ptr = reinterpret_cast<T*>(acc_bytes);
        auto policy = rmm::exec_policy(stream);

        thrust::transform(policy, acc_ptr, acc_ptr + count, in_ptr, acc_ptr, op);
    });
}

template <typename T, ReduceOp Op>
ReduceKernel make_reduce_kernel_impl() {
    if constexpr (Op == ReduceOp::SUM) {
        return [](PackedData& accum, PackedData&& incoming) {
            if constexpr (std::is_same_v<T, bool>) {
                device_elementwise_reduce<T>(
                    accum.data.get(), incoming.data.get(), thrust::logical_or<T>{}
                );
            } else {
                device_elementwise_reduce<T>(
                    accum.data.get(), incoming.data.get(), thrust::plus<T>{}
                );
            }
        };
    } else if constexpr (Op == ReduceOp::PROD) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::multiplies<T>{}
            );
        };
    } else if constexpr (Op == ReduceOp::MIN) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::minimum<T>{}
            );
        };
    } else if constexpr (Op == ReduceOp::MAX) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::maximum<T>{}
            );
        };
    } else {
        static_assert(
            Op == ReduceOp::SUM || Op == ReduceOp::PROD || Op == ReduceOp::MIN
                || Op == ReduceOp::MAX,
            "Device reduction kernel only implemented for SUM, PROD, MIN, and MAX"
        );
    }
}

}  // namespace

// Explicit specializations for the (T, Op) combinations we support on device.

template <>
ReduceKernel make_reduce_kernel<int, ReduceOp::SUM>() {
    return make_reduce_kernel_impl<int, ReduceOp::SUM>();
}

template <>
ReduceKernel make_reduce_kernel<int, ReduceOp::PROD>() {
    return make_reduce_kernel_impl<int, ReduceOp::PROD>();
}

template <>
ReduceKernel make_reduce_kernel<int, ReduceOp::MIN>() {
    return make_reduce_kernel_impl<int, ReduceOp::MIN>();
}

template <>
ReduceKernel make_reduce_kernel<int, ReduceOp::MAX>() {
    return make_reduce_kernel_impl<int, ReduceOp::MAX>();
}

template <>
ReduceKernel make_reduce_kernel<float, ReduceOp::SUM>() {
    return make_reduce_kernel_impl<float, ReduceOp::SUM>();
}

template <>
ReduceKernel make_reduce_kernel<float, ReduceOp::PROD>() {
    return make_reduce_kernel_impl<float, ReduceOp::PROD>();
}

template <>
ReduceKernel make_reduce_kernel<float, ReduceOp::MIN>() {
    return make_reduce_kernel_impl<float, ReduceOp::MIN>();
}

template <>
ReduceKernel make_reduce_kernel<float, ReduceOp::MAX>() {
    return make_reduce_kernel_impl<float, ReduceOp::MAX>();
}

template <>
ReduceKernel make_reduce_kernel<double, ReduceOp::SUM>() {
    return make_reduce_kernel_impl<double, ReduceOp::SUM>();
}

template <>
ReduceKernel make_reduce_kernel<double, ReduceOp::PROD>() {
    return make_reduce_kernel_impl<double, ReduceOp::PROD>();
}

template <>
ReduceKernel make_reduce_kernel<double, ReduceOp::MIN>() {
    return make_reduce_kernel_impl<double, ReduceOp::MIN>();
}

template <>
ReduceKernel make_reduce_kernel<double, ReduceOp::MAX>() {
    return make_reduce_kernel_impl<double, ReduceOp::MAX>();
}

template <>
ReduceKernel make_reduce_kernel<unsigned long, ReduceOp::SUM>() {
    return make_reduce_kernel_impl<unsigned long, ReduceOp::SUM>();
}

template <>
ReduceKernel make_reduce_kernel<unsigned long, ReduceOp::PROD>() {
    return make_reduce_kernel_impl<unsigned long, ReduceOp::PROD>();
}

template <>
ReduceKernel make_reduce_kernel<unsigned long, ReduceOp::MIN>() {
    return make_reduce_kernel_impl<unsigned long, ReduceOp::MIN>();
}

template <>
ReduceKernel make_reduce_kernel<unsigned long, ReduceOp::MAX>() {
    return make_reduce_kernel_impl<unsigned long, ReduceOp::MAX>();
}

}  // namespace rapidsmpf::allreduce::detail
