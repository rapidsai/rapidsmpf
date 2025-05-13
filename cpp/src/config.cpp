/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

namespace rapidsmpf::config {

Options::Options(
    std::unordered_map<std::string, std::unique_ptr<Option>> options,
    std::unordered_map<std::string, std::string> options_as_strings
)
    : options_{std::move(options)}, options_as_strings_{std::move(options_as_strings)} {}


}  // namespace rapidsmpf::config
