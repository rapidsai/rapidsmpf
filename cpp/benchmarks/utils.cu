/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/random.h>
#include <thrust/transform.h>

#include <cudf/types.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include "utils.hpp"

std::unique_ptr<cudf::column> random_column(
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    // Fill vector with random data.
    rmm::device_uvector<std::int32_t> vec(nrows, stream, mr);
    thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nrows),
        vec.begin(),
        [min_val, max_val] __device__(cudf::size_type index) {
            thrust::default_random_engine engine(index);  // HACK: use the seed as index
            thrust::uniform_int_distribution<std::int32_t> dist(min_val, max_val);
            return dist(engine);
        }
    );
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
