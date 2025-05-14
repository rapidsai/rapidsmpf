/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>

#include <unistd.h>

#include <rapidsmpf/config.hpp>

extern char** environ;

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

void get_environment_variables(
    std::unordered_map<std::string, std::string>& output, std::string const& key_regex
) {
    RAPIDSMPF_EXPECTS(
        std::regex(key_regex).mark_count() == 1,
        "key_regex must contain exactly one capture group (e.g., \"RAPIDSMPF_(.*)\")",
        std::invalid_argument
    );

    std::regex pattern(key_regex + "=(.*)");
    for (char** env = environ; *env != nullptr; ++env) {
        std::string entry(*env);
        std::smatch match;
        if (std::regex_match(entry, match, pattern)) {
            if (match.size() == 3) {  // match[1]: captured key, match[2]: value
                output.insert({match[1].str(), match[2].str()});
            }
        }
    }
}

std::unordered_map<std::string, std::string> get_environment_variables(
    std::string const& key_regex
) {
    std::unordered_map<std::string, std::string> ret;
    get_environment_variables(ret, key_regex);
    return ret;
}

}  // namespace rapidsmpf::config
