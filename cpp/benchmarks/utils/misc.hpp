/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <rapidsmp/error.hpp>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/**
 * @brief Computes the harmonic mean of a set of values.
 *
 * The harmonic mean is defined as:
 * \f[
 * \text{HM} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}
 * \f]
 * where \f$n\f$ is the number of values, and \f$x_i\f$ are the individual values.
 *
 * @param values A vector of double values for which the harmonic mean is to be computed.
 *               All values must be non-zero, and the vector must not be empty.
 * @return The harmonic mean as a `double`.
 *
 * @throws std::invalid_argument If the input vector is empty.
 * @throws std::domain_error If any value in the input vector is zero.
 */
double harmonic_mean(std::vector<double> const& values) {
    if (values.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }
    double sum = 0.0;
    for (double value : values) {
        if (value == 0.0) {
            throw std::domain_error("Cannot compute harmonic mean with zero values.");
        }
        sum += 1.0 / value;
    }
    return static_cast<double>(values.size()) / sum;
}

/**
 * @brief Parses a string into an integer and ensures it is within the specified range.
 *
 * This function converts a string representation of an integer into a value of type T.
 * It performs range validation using the provided minimum and maximum values.
 *
 * @tparam T The target integer type.
 * @param output Reference to the output variable where the parsed integer will be stored.
 * @param str The input string representing the integer.
 * @param min_val The minimum allowed value.
 * @param max_val The maximum allowed value.
 * @throws std::invalid_argument If the string is not a valid integer.
 * @throws std::out_of_range If the parsed value is outside the specified range.
 */
template <typename T>
void parse_integer(
    T& output,
    std::string const& str,
    std::int64_t min_val = 0,
    std::int64_t max_val = std::numeric_limits<std::int64_t>::max()
) {
    long long val;
    try {
        val = std::stoll(str);
    } catch (std::invalid_argument const&) {
        RAPIDSMP_FAIL("cannot parse \"" + str + "\"", std::invalid_argument);
    } catch (std::out_of_range const&) {
        RAPIDSMP_FAIL("\"" + str + "\" is out of range", std::out_of_range);
    }
    RAPIDSMP_EXPECTS(min_val <= val && val <= max_val, "\"" + str + "\" is out of range");
    output = static_cast<T>(val);
}
