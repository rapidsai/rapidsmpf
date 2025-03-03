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

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rapidsmp {

/**
 * @brief Retrieves the value of an environment variable and converts it to a specified
 * type.
 *
 * If the environment variable is not set, the function returns a specified default value.
 *
 * @tparam T The type to convert the environment variable to. Must support input stream
 * extraction.
 * @param env_var_name Name of the environment variable to retrieve.
 * @param default_val Default value to return if the environment variable is not set.
 * @return The value of the environment variable converted to type `T`, or `default_val`
 * if the variable is not set.
 *
 * @throws std::invalid_argument If the value of the environment variable cannot be
 * converted to the specified type `T`.
 */
template <typename T>
T getenv_or(std::string const& env_var_name, T default_val) {
    auto const* env_val = std::getenv(env_var_name.c_str());
    if (env_val == nullptr) {
        return default_val;
    }

    std::stringstream sstream(env_val);
    T converted_val;
    sstream >> converted_val;
    if (sstream.fail()) {
        throw std::invalid_argument(
            "unknown config value " + std::string{env_var_name} + "="
            + std::string{env_val}
        );
    }
    return converted_val;
}

/**
 * @brief Specialization of `getenv_or` for boolean values.
 *
 * Converts the value of the environment variable to a boolean. If the environment
 * variable is not set, a specified default value is returned. This function handles
 * common boolean representations such as `true`, `false`, `on`, `off`, `yes`, and `no`,
 * as well as numeric representations (e.g., `0` or `1`).
 *
 * @param env_var_name Name of the environment variable to retrieve.
 * @param default_val Default value to return if the environment variable is not set.
 * @return The boolean value of the environment variable, or `default_val` if the variable
 * is not set.
 *
 * @throws std::invalid_argument If the value of the environment variable cannot be
 * interpreted as a boolean.
 */
template <>
bool getenv_or(std::string const& env_var_name, bool default_val);

}  // namespace rapidsmp
