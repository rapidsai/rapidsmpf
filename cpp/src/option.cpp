/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmp/option.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp {

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

    std::string str = to_lower(trim(env_val));
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
