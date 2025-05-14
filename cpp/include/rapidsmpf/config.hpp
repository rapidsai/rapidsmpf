/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::config {

/**
 * @brief Base class for configuration options.
 *
 * All configuration options must derive from this class.
 */
class Option {
  public:
    /**
     * @brief Virtual destructor for the `Option` base class.
     */
    virtual ~Option() = default;
};

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
        std::unordered_map<std::string, std::unique_ptr<Option>> options
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * Refer to the `Options::get` method for usage details.
     *
     * @tparam T The type of the option to retrieve. Must derive from `Option`.
     * @param key The key of the option to retrieve.
     * @return A pointer to the retrieved option.
     * @throws std::invalid_argument If the option exists but is of an incompatible type.
     */
    template <typename T>
    T const* get(std::string const& key) {
        static_assert(std::is_base_of<Option, T>::value, "T must derive from Option");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (options_.find(key) == options_.end()) {
                if (options_as_strings_.find(key) == options_as_strings_.end()) {
                    options_[key] = std::make_unique<T>();
                } else {
                    options_[key] = std::make_unique<T>(options_as_strings_[key]);
                }
            }
        }
        auto option = dynamic_cast<T*>(options_[key].get());
        RAPIDSMPF_EXPECTS(
            option != nullptr,
            "accessing option with incompatible template type",
            std::invalid_argument
        );
        return option;
    }

  private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> options_as_strings_;
    std::unordered_map<std::string, std::unique_ptr<Option>> options_;
};

}  // namespace detail

/**
 * @brief Manages configuration options for the RapidsMPF operations.
 *
 * The `Options` class provides a high-level interface for storing and retrieving
 * configuration options.
 *
 * All keys are trimmed and converted to lower case using `rapidsmpf::trim()` and
 * `rapidsmpf::to_lower()`.
 *
 * @note Copying `rapidsmpf::Options` is fast as it uses a shared pointer to its internal
 * implementation (`OptionsImpl`).
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
        std::unordered_map<std::string, std::unique_ptr<Option>> options = {}
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * If the option does not exist, it is created using:
     *  1) its string representation if available, or
     *  2) using its default constructor.
     *
     * @tparam T The type of the option to retrieve. Must derive from `Option`.
     * @param key The key of the option to retrieve.
     * @return A pointer to the retrieved option.
     * @throws std::invalid_argument If the option exists but is of an incompatible type.
     */
    template <typename T>
    T const* get(std::string const& key) {
        return impl_->get<T>(key);
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
