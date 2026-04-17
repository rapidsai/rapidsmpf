/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <string>

namespace rapidsmpf::rrun {

/**
 * @brief RAII guard that saves, optionally modifies, and restores an environment
 * variable.
 *
 * On construction the current value of the named variable is captured. The
 * caller may pass an initial value to set, or `nullptr` to unset the variable.
 * On destruction the original state is restored unconditionally, making this
 * class safe to use across code that may throw.
 *
 * @code
 * {
 *     ScopedEnvVar guard("CUDA_VISIBLE_DEVICES", nullptr);  // unset
 *     // ... topology discovery sees all GPUs ...
 * }  // original CUDA_VISIBLE_DEVICES restored here
 * @endcode
 */
class ScopedEnvVar {
  public:
    /**
     * @brief Construct the guard, saving and optionally replacing the variable.
     *
     * @param name Name of the environment variable.
     * @param value Value to set, or `nullptr` to unset the variable.
     */
    ScopedEnvVar(char const* name, char const* value) : name_(name) {
        char const* old = std::getenv(name);
        if (old != nullptr) {
            had_value_ = true;
            old_value_ = old;
        }
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(ScopedEnvVar const&) = delete;
    ScopedEnvVar& operator=(ScopedEnvVar const&) = delete;

  private:
    std::string name_;
    std::string old_value_;
    bool had_value_{false};
};

}  // namespace rapidsmpf::rrun
