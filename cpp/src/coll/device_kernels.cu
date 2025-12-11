/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <concepts>
#include <type_traits>

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <rmm/exec_policy.hpp>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>

namespace rapidsmpf::coll::detail {

using ReduceOperatorFunction = rapidsmpf::coll::ReduceOperatorFunction;

namespace {

template <typename T, typename Op>
void device_elementwise_reduce(Buffer* acc_buf, Buffer* in_buf, Op op) {
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
        acc_nbytes % sizeof(T) == 0,
        "AllReduce device reduction buffer size must be multiple of sizeof(T)"
    );

    auto const count = acc_nbytes / sizeof(T);

    // Ensure the accumulator stream waits for the incoming buffer's latest write.
    cuda_stream_join(acc_buf->stream(), in_buf->stream());

    // Destination buffer: use write_access so the latest-write event is updated.
    acc_buf->write_access([&](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
        auto* acc_ptr = reinterpret_cast<T*>(acc_bytes);
        auto policy = rmm::exec_policy_nosync(stream);

        // Safe to read from incoming buffer after the stream join above.
        auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());
        auto const* in_ptr = reinterpret_cast<T const*>(in_bytes);

        thrust::transform(policy, acc_ptr, acc_ptr + count, in_ptr, acc_ptr, op);
    });
}

template <typename T, DeviceReduceOp Op>
    requires ValidDeviceReduceOp<T, Op>
ReduceOperatorFunction make_reduce_operator_impl() {
    if constexpr (Op == DeviceReduceOp::SUM) {
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
    } else if constexpr (Op == DeviceReduceOp::PROD) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::multiplies<T>{}
            );
        };
    } else if constexpr (Op == DeviceReduceOp::MIN) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::minimum<T>{}
            );
        };
    } else if constexpr (Op == DeviceReduceOp::MAX) {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::maximum<T>{}
            );
        };
    } else {
        static_assert(
            Op == DeviceReduceOp::SUM || Op == DeviceReduceOp::PROD
                || Op == DeviceReduceOp::MIN || Op == DeviceReduceOp::MAX,
            "Unsupported DeviceReduceOp"
        );
    }
}

}  // namespace

template <typename T, DeviceReduceOp Op>
    requires ValidDeviceReduceOp<T, Op>
ReduceOperatorFunction make_device_reduce_operator_impl() {
    return make_reduce_operator_impl<T, Op>();
}

#define RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(T)                   \
    template ReduceOperatorFunction                              \
    make_device_reduce_operator_impl<T, DeviceReduceOp::SUM>();  \
    template ReduceOperatorFunction                              \
    make_device_reduce_operator_impl<T, DeviceReduceOp::PROD>(); \
    template ReduceOperatorFunction                              \
    make_device_reduce_operator_impl<T, DeviceReduceOp::MIN>();  \
    template ReduceOperatorFunction                              \
    make_device_reduce_operator_impl<T, DeviceReduceOp::MAX>();

RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(bool)
RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(int)
RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(float)
RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(double)
RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE(unsigned long)

#undef RAPIDSMPF_INSTANTIATE_DEVICE_REDUCE

}  // namespace rapidsmpf::coll::detail
