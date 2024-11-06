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

#include <chrono>
#include <sstream>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/**
 * @brief Generates a random numeric column (std::int32_t).
 *
 * Creates a cuDF column with random integer values uniformly distributed in the range
 * `[min_val, max_val]`.
 *
 * @param nrows Number of rows in the generated column.
 * @param min_val Minimum value (inclusive) for the random data.
 * @param max_val Maximum value (inclusive) for the random data.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param mr Device memory resource for allocating the column.
 * @return A unique pointer to the generated cuDF column.
 *
 * @note The function uses the specified CUDA stream for asynchronous operations.
 */
std::unique_ptr<cudf::column> random_column(
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);

/**
 * @brief Generates a random numeric table (std::int32_t).
 *
 * Creates a cuDF table consisting of multiple columns with random integer values, each
 * uniformly distributed in the range `[min_val, max_val]`.
 *
 * @param ncolumns Number of columns in the generated table.
 * @param nrows Number of rows in each column of the table.
 * @param min_val Minimum value (inclusive) for the random data.
 * @param max_val Maximum value (inclusive) for the random data.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param mr Device memory resource for allocating the table.
 * @return A cuDF table containing the generated random columns.
 *
 * @note Each column in the table will have the same number of rows and data distribution.
 */
cudf::table random_table(
    cudf::size_type ncolumns,
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);

template <typename T>
std::string to_precision(T value, int precision = 2) {
    std::stringstream ss;
    ss.precision(precision);
    ss << std::fixed;
    ss << value;
    return ss.str();
}

std::string inline to_precision(Duration value, int precision = 2) {
    return to_precision(value.count(), precision);
}

std::string inline to_mib(double nbytes, int precision = 2) {
    return to_precision(nbytes / (1 << 20), precision);
}
