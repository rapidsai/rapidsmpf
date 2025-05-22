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

Options::Options(std::unordered_map<std::string, OptionValue> options)
    : shared_{std::make_shared<detail::SharedOptions>()} {
    // insert, trim and lower case all keys.
    auto& opts = shared_->options;
    opts.reserve(options.size());
    for (auto&& [key, value] : options) {
        auto new_key = rapidsmpf::to_lower(rapidsmpf::trim(key));
        RAPIDSMPF_EXPECTS(
            opts.emplace(std::move(new_key), std::move(value)).second,
            "option keys must be case-insensitive",
            std::invalid_argument
        );
    }
}

namespace {
// Helper function to get OptionValue map from options-as-strings map.
std::unordered_map<std::string, OptionValue> from_options_as_strings(
    std::unordered_map<std::string, std::string>&& options_as_strings
) {
    std::unordered_map<std::string, OptionValue> ret;
    for (auto&& [key, val] : options_as_strings) {
        ret.emplace(std::move(key), OptionValue(std::move(val)));
    }
    return ret;
}
}  // namespace

Options::Options(std::unordered_map<std::string, std::string> options_as_strings)
    : Options(from_options_as_strings(std::move(options_as_strings))){};

std::unordered_map<std::string, std::string> Options::get_strings() const {
    auto const& shared = *shared_;
    std::unordered_map<std::string, std::string> ret;
    std::lock_guard<std::mutex> lock(shared.mutex);
    for (const auto& [key, option] : shared.options) {
        ret[key] = option.get_value_as_string();
    }
    return ret;
}

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
