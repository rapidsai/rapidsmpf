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
#pragma once

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace rapidsmp {

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

/**
 * @brief Formats value to a string with a specified number of decimal places.
 *
 * @tparam T The type of the value to format.
 * @param value The value to format.
 * @param precision The number of decimal places to include.
 * @return A string representation of the value with the specified precision.
 */
template <typename T>
std::string to_precision(T value, int precision = 2) {
    std::stringstream ss;
    ss.precision(precision);
    ss << std::fixed;
    ss << value;
    return ss.str();
}

/**
 * @brief Format number of bytes to a human readable string representation.
 *
 * @param nbytes The number of bytes to convert.
 * @param precision The number of decimal places to include.
 * @return A string representation of the byte size with the specified precision.
 */
std::string inline format_nbytes(int64_t nbytes, int precision = 2) {
    constexpr std::array<const char*, 6> units = {" B", " KiB", " MiB", " GiB", " TiB"};
    auto n = static_cast<double>(nbytes);
    for (auto const& unit : units) {
        if (std::abs(n) < 1024.0) {
            return to_precision(n, precision) + unit;
        }
        n /= 1024.0;
    }
    return to_precision(n, precision) + " PiB";
}

/**
 * @brief Extracts the value associated with a specific key from a map, removing the
 * key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @tparam KeyType The type of the key.
 * @param map The map from which to extract the value.
 * @param key The key associated with the value to extract.
 * @return The extracted value.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType, typename KeyType>
typename MapType::mapped_type extract_value(MapType& map, KeyType const& key) {
    auto node = map.extract(key);
    if (!node) {
        throw std::out_of_range("key not found");
    }
    return std::move(node.mapped());
}

/**
 * @brief Extracts a key from a map, removing the key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @tparam KeyType The type of the key.
 * @param map The map from which to extract the key.
 * @param key The key to extract.
 * @return The extracted key.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType, typename KeyType>
typename MapType::key_type extract_key(MapType& map, KeyType const& key) {
    auto node = map.extract(key);
    if (!node) {
        throw std::out_of_range("key not found");
    }
    return std::move(node.key());
}

/**
 * @brief Extracts a key-value pair from a map, removing it from the map.
 *
 * @tparam MapType The type of the associative container.
 * @tparam KeyType The type of the key.
 * @param map The map from which to extract the key-value pair.
 * @param key The key associated with the pair to extract.
 * @return A pair containing the extracted key and value.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType, typename KeyType>
std::pair<typename MapType::key_type, typename MapType::mapped_type> extract_item(
    MapType& map, KeyType const& key
) {
    auto node = map.extract(key);
    if (!node) {
        throw std::out_of_range("key not found");
    }
    return {std::move(node.key()), std::move(node.mapped())};
}

/**
 * @brief Checks whether the application is running under Valgrind.
 *
 * @return `true` if the application is running under Valgrind, `false` otherwise.
 */
bool is_running_under_valgrind();

}  // namespace rapidsmp
