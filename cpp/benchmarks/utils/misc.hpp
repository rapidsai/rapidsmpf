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

#include <chrono>
#include <sstream>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

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
 * @brief Formats a duration to a string with a specified number of decimal places.
 *
 * @param value The duration to format.
 * @param precision The number of decimal places to include.
 * @return A string representation of the duration with the specified precision.
 */
std::string inline to_precision(Duration value, int precision = 2) {
    return to_precision(value.count(), precision);
}

/**
 * @brief Converts number of bytes to a string representation in MiB.
 *
 * @param nbytes The number of bytes to convert.
 * @param precision The number of decimal places to include.
 * @return A string representation of the byte size in MiB with the specified precision.
 */
std::string inline to_mib(double nbytes, int precision = 2) {
    return to_precision(nbytes / (1 << 20), precision);
}
