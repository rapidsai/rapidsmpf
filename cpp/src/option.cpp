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

#include <algorithm>

#include <rapidsmp/option.hpp>

namespace rapidsmp {

namespace detail {
std::string trim_and_lowercase(std::string str) {
    // Special considerations regarding the case conversion:
    // - std::tolower() is not an addressable function. Passing it to std::transform()
    //   as a function pointer, if the compile turns out successful, causes the program
    //   behavior "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is
    //   addressable and does not have this problem, but the following item still applies.
    // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to
    // unsigned char.
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    // Trim whitespaces
    std::stringstream trimmer;
    trimmer << str;
    str.clear();
    trimmer >> str;
    return str;
}
}  // namespace detail

template <>
bool getenv_or(std::string const& env_var_name, bool default_val) {
    auto const* env_val = std::getenv(env_var_name.c_str());
    if (env_val == nullptr) {
        return default_val;
    }
    try {
        // Try parsing `env_var_name` as a integer
        return static_cast<bool>(std::stoi(env_val));
    } catch (std::invalid_argument const&) {
    }

    std::string str = detail::trim_and_lowercase(env_val);
    if (str == "true" || str == "on" || str == "yes") {
        return true;
    }
    if (str == "false" || str == "off" || str == "no") {
        return false;
    }
    throw std::invalid_argument(
        "unknown config value " + std::string{env_var_name} + "=" + std::string{env_val}
    );
}


}  // namespace rapidsmp
