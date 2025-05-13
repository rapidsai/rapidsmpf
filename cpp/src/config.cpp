/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

namespace rapidsmpf::config {

namespace detail {

OptionsImpl::OptionsImpl(
    std::unordered_map<std::string, std::string> options_as_strings,
    std::unordered_map<std::string, std::unique_ptr<Option>> options

)
    : options_as_strings_{std::move(options_as_strings)}, options_{std::move(options)} {}

}  // namespace detail

Options::Options(
    std::unordered_map<std::string, std::string> options_as_strings,
    std::unordered_map<std::string, std::unique_ptr<Option>> options

)
    : impl_{std::make_shared<detail::OptionsImpl>(
        std::move(options_as_strings), std::move(options)
    )} {}


}  // namespace rapidsmpf::config
