/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::config {

/**
 * @brief Type alias for a factory function that constructs options from strings.
 *
 * The factory receives the string representation of an option value and returns
 * an instance of the option type. If the option is unset, the function receives
 * an empty string and should either return a meaningful default value or throw
 * `std::invalid_argument`.
 *
 * @note The factory must not access other options, as a lock is held during option
 * initialization and doing so may cause a deadlock.
 */
template <typename T>
using OptionFactory = std::function<T(std::string const&)>;

namespace detail {

/**
 * @brief Internal implementation of the `Options` class.
 *
 * This class is used internally by `Options` to manage the storage and retrieval
 * of configuration options. Refer to the `Options` class documentation for details.
 */
class OptionsImpl {
  public:
    /**
     * @brief Constructs an `OptionsImpl` instance.
     *
     * @param options_as_strings A map of option keys to their string representations.
     * @param options A map of option keys to their corresponding `Option` objects.
     */
    OptionsImpl(
        std::unordered_map<std::string, std::string> options_as_strings,
        std::unordered_map<std::string, std::any> options
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * Refer to the `Options::get` method for usage details.
     *
     * @tparam T The type of the option to retrieve.
     * @param key The option key (should be lower case).
     * @param factory Function to construct the option from a string.
     * @return Reference to the option value.
     * @throws std::invalid_argument If the stored option type does not match T.
     */
    template <typename T>
    T const& get(std::string const& key, OptionFactory<T> factory) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (options_.find(key) == options_.end()) {
                if (options_as_strings_.find(key) == options_as_strings_.end()) {
                    options_[key] = std::make_any<T>(factory(""));
                } else {
                    options_[key] = std::make_any<T>(factory(options_as_strings_[key]));
                }
            }
        }
        try {
            return std::any_cast<T const&>(options_[key]);
        } catch (std::bad_any_cast const&) {
            RAPIDSMPF_FAIL(
                "accessing option with incompatible template type", std::invalid_argument
            );
        }
    }

  private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> options_as_strings_;
    std::unordered_map<std::string, std::any> options_;
};
}  // namespace detail

/**
 * @brief Manages configuration options for RapidsMPF operations.
 *
 * The `Options` class provides a high-level interface for storing and retrieving
 * configuration options.
 *
 * All keys are trimmed and converted to lower case using `rapidsmpf::trim()` and
 * `rapidsmpf::to_lower()`.
 *
 * @note Copying `rapidsmpf::config::Options` is efficient as it uses a shared pointer
 * to its internal implementation (`OptionsImpl`).
 */
class Options {
  public:
    /**
     * @brief Constructs an `Options` instance.
     *
     * @param options_as_strings A map of option keys to their string representations.
     * @param options A map of option keys to their corresponding `Option` objects.
     */
    Options(
        std::unordered_map<std::string, std::string> options_as_strings = {},
        std::unordered_map<std::string, std::any> options = {}
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * If the option is not present, it will be constructed using the provided
     * factory, which receives the string representation of the option (or an
     * empty string if unset).
     *
     * @tparam T The type of the option to retrieve.
     * @param key The option key (should be lower case).
     * @param factory Function to construct the option from a string.
     * @return Reference to the option value.
     * @throws std::invalid_argument If the stored option type does not match T.
     */
    template <typename T>
    T const& get(std::string const& key, OptionFactory<T> factory) {
        return impl_->get<T>(key, factory);
    }

  private:
    std::shared_ptr<detail::OptionsImpl> impl_;
};

/**
 * @brief Populates a map with environment variables matching a given regular expression.
 *
 * This function scans the current process's environment variables and inserts those whose
 * keys match the provided regular expression into the `output` map. Only variables with
 * keys not already present in `output` are inserted; existing keys are left unchanged.
 *
 * The `key_regex` should contain a single capture group that extracts the portion of the
 * environment variable key you want to use as the map key. For example, to strip the
 * `RAPIDSMPF_` prefix, use `RAPIDSMPF_(.*)` as the regex. The captured group will be used
 * as the key in the output map.
 *
 * Example:
 *   - Environment variable: RAPIDSMPF_FOO=bar
 *   - key_regex: "RAPIDSMPF_(.*)"
 *   - Resulting map entry: { "FOO", "bar" }
 *
 * @param[out] output The map to populate with matching environment variables. Only keys
 * that do not already exist in the map will be added.
 * @param[in] key_regex A regular expression with a single capture group to match and
 * extract the environment variable keys. Only environment variables with keys matching
 * this pattern will be considered.
 * @throws std::invalid_argument If key_regex doesn't contain exactly one capture group.
 *
 * @warning This function uses `std::regex` and relies on the global `environ` symbol,
 * which is POSIX-specific and is **not** thread-safe.
 */
void get_environment_variables(
    std::unordered_map<std::string, std::string>& output,
    std::string const& key_regex = "RAPIDSMPF_(.*)"
);

/**
 * @brief Returns a map of environment variables matching a given regular expression.
 *
 * This is a convenience overload. See the documentation for the first variant of
 * `get_environment_variables()` for details on matching and behavior.
 *
 * @param key_regex A regular expression with a single capture group to match and extract
 * the environment variable keys.
 * @return A map containing all matching environment variables, with keys as extracted by
 * the capture group.
 * @throws std::invalid_argument If key_regex doesn't contain exactly one capture group.
 *
 * @see get_environment_variables(std::unordered_map<std::string, std::string>&,
 * std::string const&)
 */
std::unordered_map<std::string, std::string> get_environment_variables(
    std::string const& key_regex = "RAPIDSMPF_(.*)"
);

}  // namespace rapidsmpf::config
