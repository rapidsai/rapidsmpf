/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <rmm/exec_policy.hpp>

#include <rapidsmpf/coll/allreduce.hpp>
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

    // Input buffer: assume ready as provided by the communicator / AllReduce.
    auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());
    auto const* in_ptr = reinterpret_cast<T const*>(in_bytes);

    // Destination buffer: use write_access so the latest-write event is updated.
    acc_buf->write_access([&](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
        auto* acc_ptr = reinterpret_cast<T*>(acc_bytes);
        auto policy = rmm::exec_policy_nosync(stream);

        thrust::transform(policy, acc_ptr, acc_ptr + count, in_ptr, acc_ptr, op);
    });
}

template <typename T, DeviceReduceOp Op>
struct ReduceOperatorMaker;

template <typename T>
struct ReduceOperatorMaker<T, DeviceReduceOp::SUM> {
    static ReduceOperatorFunction make() {
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
    }
};

template <typename T>
struct ReduceOperatorMaker<T, DeviceReduceOp::PROD> {
    static ReduceOperatorFunction make() {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::multiplies<T>{}
            );
        };
    }
};

template <typename T>
struct ReduceOperatorMaker<T, DeviceReduceOp::MIN> {
    static ReduceOperatorFunction make() {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::minimum<T>{}
            );
        };
    }
};

template <typename T>
struct ReduceOperatorMaker<T, DeviceReduceOp::MAX> {
    static ReduceOperatorFunction make() {
        return [](PackedData& accum, PackedData&& incoming) {
            device_elementwise_reduce<T>(
                accum.data.get(), incoming.data.get(), thrust::maximum<T>{}
            );
        };
    }
};

template <typename T, DeviceReduceOp Op>
ReduceOperatorFunction make_reduce_operator_impl() {
    return ReduceOperatorMaker<T, Op>::make();
}

}  // namespace

// Explicit specializations for the (T, Op) combinations we support on device.

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<int, DeviceReduceOp::SUM>() {
    return make_reduce_operator_impl<int, DeviceReduceOp::SUM>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<int, DeviceReduceOp::PROD>() {
    return make_reduce_operator_impl<int, DeviceReduceOp::PROD>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<int, DeviceReduceOp::MIN>() {
    return make_reduce_operator_impl<int, DeviceReduceOp::MIN>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<int, DeviceReduceOp::MAX>() {
    return make_reduce_operator_impl<int, DeviceReduceOp::MAX>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<float, DeviceReduceOp::SUM>() {
    return make_reduce_operator_impl<float, DeviceReduceOp::SUM>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<float, DeviceReduceOp::PROD>() {
    return make_reduce_operator_impl<float, DeviceReduceOp::PROD>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<float, DeviceReduceOp::MIN>() {
    return make_reduce_operator_impl<float, DeviceReduceOp::MIN>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<float, DeviceReduceOp::MAX>() {
    return make_reduce_operator_impl<float, DeviceReduceOp::MAX>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<double, DeviceReduceOp::SUM>() {
    return make_reduce_operator_impl<double, DeviceReduceOp::SUM>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<double, DeviceReduceOp::PROD>() {
    return make_reduce_operator_impl<double, DeviceReduceOp::PROD>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<double, DeviceReduceOp::MIN>() {
    return make_reduce_operator_impl<double, DeviceReduceOp::MIN>();
}

template <>
ReduceOperatorFunction make_device_reduce_operator_impl<double, DeviceReduceOp::MAX>() {
    return make_reduce_operator_impl<double, DeviceReduceOp::MAX>();
}

template <>
ReduceOperatorFunction
make_device_reduce_operator_impl<unsigned long, DeviceReduceOp::SUM>() {
    return make_reduce_operator_impl<unsigned long, DeviceReduceOp::SUM>();
}

template <>
ReduceOperatorFunction
make_device_reduce_operator_impl<unsigned long, DeviceReduceOp::PROD>() {
    return make_reduce_operator_impl<unsigned long, DeviceReduceOp::PROD>();
}

template <>
ReduceOperatorFunction
make_device_reduce_operator_impl<unsigned long, DeviceReduceOp::MIN>() {
    return make_reduce_operator_impl<unsigned long, DeviceReduceOp::MIN>();
}

template <>
ReduceOperatorFunction
make_device_reduce_operator_impl<unsigned long, DeviceReduceOp::MAX>() {
    return make_reduce_operator_impl<unsigned long, DeviceReduceOp::MAX>();
}

}  // namespace rapidsmpf::coll::detail
