/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>

#include <unistd.h>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils.hpp>

extern char** environ;

namespace rapidsmpf::config {

namespace detail {

namespace {
// Helper function to trim and lower case all keys in a map.
template <typename T>
std::unordered_map<std::string, T> transform_keys_trim_lower(
    std::unordered_map<std::string, T>&& input
) {
    std::unordered_map<std::string, T> ret;
    ret.reserve(input.size());
    for (auto&& [key, value] : input) {
        auto new_key = rapidsmpf::to_lower(rapidsmpf::trim(key));
        RAPIDSMPF_EXPECTS(
            ret.emplace(std::move(new_key), std::move(value)).second,
            "keys must be case-insensitive",
            std::invalid_argument
        );
    }
    return ret;
}

// Helper function to get OptionValue map from options-as-strings map.
std::unordered_map<std::string, OptionValue> from_options_as_strings(
    std::unordered_map<std::string, std::string>&& options_as_strings
) {
    std::unordered_map<std::string, OptionValue> ret;
    for (auto&& [key, val] : transform_keys_trim_lower(std::move(options_as_strings))) {
        ret.insert({std::move(key), OptionValue(std::move(val))});
    }
    return ret;
}
}  // namespace

OptionsImpl::OptionsImpl(std::unordered_map<std::string, OptionValue> options)
    : options_{transform_keys_trim_lower(std::move(options))} {}

}  // namespace detail

Options::Options(std::unordered_map<std::string, OptionValue> options)
    : impl_{std::make_shared<detail::OptionsImpl>(std::move(options))} {}

Options::Options(std::unordered_map<std::string, std::string> options_as_strings)
    : Options(detail::from_options_as_strings(std::move(options_as_strings))){};

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
