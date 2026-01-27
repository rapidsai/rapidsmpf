/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/random.h>
#include <thrust/transform.h>

#include <cudf/types.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include "random_data.hpp"

rmm::device_uvector<std::int32_t> random_device_vector(
    cudf::size_type nelem,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    // Fill vector with random data.
    rmm::device_uvector<std::int32_t> vec(static_cast<std::size_t>(nelem), stream, mr);
    thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nelem),
        vec.begin(),
        [min_val, max_val] __device__(cudf::size_type index) {
            thrust::default_random_engine engine(index);  // HACK: use the seed as index
            thrust::uniform_int_distribution<std::int32_t> dist(min_val, max_val);
            return dist(engine);
        }
    );
    return vec;
}

std::unique_ptr<cudf::column> random_column(
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    auto vec = random_device_vector(nrows, min_val, max_val, stream, mr);
    return std::make_unique<cudf::column>(
        std::move(vec), rmm::device_buffer{0, stream, mr}, 0
    );
}

cudf::table random_table(
    cudf::size_type ncolumns,
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (auto i = 0; i < ncolumns; ++i) {
        cols.push_back(random_column(nrows, min_val, max_val, stream, mr));
    }
    return cudf::table(std::move(cols));
}

void random_fill(rapidsmpf::Buffer& buffer, rmm::device_async_resource_ref mr) {
    switch (buffer.mem_type()) {
    case rapidsmpf::MemoryType::DEVICE:
        {
            auto vec = random_device_vector(
                buffer.size / sizeof(std::int32_t) + sizeof(std::int32_t),
                std::numeric_limits<std::int32_t>::min(),
                std::numeric_limits<std::int32_t>::max(),
                buffer.stream(),
                mr
            );
            buffer.write_access([&](std::byte* buffer_data,
                                    rmm::cuda_stream_view stream) {
                RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                    buffer_data, vec.data(), buffer.size, cudaMemcpyDefault, stream
                ));
            });
            break;
        }
    default:
        RAPIDSMPF_FAIL("unsupported memory type", std::invalid_argument);
    }
}
