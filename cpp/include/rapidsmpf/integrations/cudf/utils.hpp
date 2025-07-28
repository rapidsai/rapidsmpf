/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <string>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace rapidsmpf {

/**
 * @brief Converts the element at a specific index in a `cudf::column_view` to a string.
 *
 * @param col The column view containing the data.
 * @param index The index of the element to convert.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of the element at the specified index.
 */
std::string str(
    cudf::column_view col,
    cudf::size_type index,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);

/**
 * @brief Converts all elements in a `cudf::column_view` to a string.
 *
 * @param col The column view containing the data.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of all elements in the column.
 */
std::string str(
    cudf::column_view col,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);

/**
 * @brief Converts all rows in a `cudf::table_view` to a string.
 *
 * @param tbl The table view containing the data.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of all rows in the table.
 */
std::string str(
    cudf::table_view tbl,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);

}  // namespace rapidsmpf
