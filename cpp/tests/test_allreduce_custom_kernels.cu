/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "test_allreduce_custom_type.hpp"

using rapidsmpf::MemoryType;
using rapidsmpf::PackedData;
using rapidsmpf::coll::ReduceOperator;

ReduceOperator make_custom_value_reduce_operator_device() {
    return [](PackedData& accum, PackedData&& incoming) {
        RAPIDSMPF_EXPECTS(
            accum.data && incoming.data,
            "CustomValue reduction operator requires non-null data buffers"
        );

        auto* acc_buf = accum.data.get();
        auto* in_buf = incoming.data.get();

        auto const acc_nbytes = acc_buf->size;
        auto const in_nbytes = in_buf->size;
        RAPIDSMPF_EXPECTS(
            acc_nbytes == in_nbytes,
            "CustomValue reduction operator requires equal-sized buffers"
        );
        RAPIDSMPF_EXPECTS(
            acc_nbytes % sizeof(CustomValue) == 0,
            "CustomValue reduction operator requires buffer size to be a multiple "
            "of sizeof(CustomValue)"
        );

        RAPIDSMPF_EXPECTS(
            acc_buf->mem_type() == MemoryType::DEVICE
                && in_buf->mem_type() == MemoryType::DEVICE,
            "CustomValue reduction operator expects device-backed buffers"
        );

        auto const count = acc_nbytes / sizeof(CustomValue);

        auto const* in_bytes = reinterpret_cast<std::byte const*>(in_buf->data());
        auto const* in_ptr = reinterpret_cast<CustomValue const*>(in_bytes);

        // Launch a device-side elementwise reduction using Thrust.
        acc_buf->write_access(
            [&, count](std::byte* acc_bytes, rmm::cuda_stream_view stream) {
                auto* acc_ptr = reinterpret_cast<CustomValue*>(acc_bytes);
                auto policy = thrust::cuda::par.on(stream.value());

                thrust::transform(
                    policy,
                    acc_ptr,
                    acc_ptr + count,
                    in_ptr,
                    acc_ptr,
                    [] __device__(CustomValue a, CustomValue b) {
                        CustomValue out;
                        out.value = a.value + b.value;
                        out.weight = a.weight < b.weight ? a.weight : b.weight;
                        return out;
                    }
                );
            }
        );
    };
}
