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

/**
 * @brief Configuration option value.
 *
 * The OptionValue class encapsulates a value (of any type using std::any)
 * and an optional string representation of the value.
 */
class OptionValue {
  public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty OptionValue.
     */
    OptionValue() = default;

    /**
     * @brief Constructs OptionValue from a std::any value.
     *
     * @param value The value to store, wrapped in std::any.
     */
    OptionValue(std::any value) : value_{std::move(value)} {}

    /**
     * @brief Constructs OptionValue from a string representation.
     *
     * @param value_as_string A string representation of the value.
     */
    OptionValue(std::string value_as_string)
        : value_as_string_{std::move(value_as_string)} {}

    /**
     * @brief Convenience constructor to store any type.
     *
     * Wraps the given value in std::any and stores it.
     *
     * @tparam T The type of the value.
     * @param value The value to store.
     */
    template <typename T>
    OptionValue(T value) : OptionValue(std::make_any<T>(value)) {}

    /**
     * @brief Retrieves the stored value.
     *
     * @return A const reference to the std::any value.
     */
    [[nodiscard]] std::any const& get_value() const {
        return value_;
    }

    /**
     * @brief Retrieves the string representation of the value.
     *
     * Is the empty string, if not string representation exist.
     *
     * @return A const reference to the string representation.
     */
    [[nodiscard]] std::string const& get_value_as_string() const {
        return value_as_string_;
    }

    /**
     * @brief Sets the value if it has not been set already.
     *
     * @param value The new value to store.
     *
     * @throws std::invalid_argument if the value is already set.
     */
    void set_value(std::any value) {
        RAPIDSMPF_EXPECTS(
            !value_.has_value(), "value already set", std::invalid_argument
        );
        value_ = std::move(value);
    }

  private:
    std::any value_{};
    std::string value_as_string_{};
    // TODO: add a collective policy.
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
     * @param options A map of option keys to their corresponding option value.
     */
    OptionsImpl(std::unordered_map<std::string, OptionValue> options);

    /**
     * @brief Retrieves a configuration option by key.
     *
     * Refer to the `Options::get` method for usage details.
     *
     * @tparam T The type of the option to retrieve.
     * @param key The option key (should be lower case).
     * @param factory Function to construct the option from a string.
     * @return Reference to the option value.
     *
     * @throws std::invalid_argument If the stored option type does not match T.
     */
    template <typename T>
    T const& get(const std::string& key, OptionFactory<T> factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& option = options_[key];
        if (!option.get_value().has_value()) {
            option.set_value(std::make_any<T>(factory(option.get_value_as_string())));
        }
        try {
            return std::any_cast<const T&>(option.get_value());
        } catch (const std::bad_any_cast&) {
            RAPIDSMPF_FAIL(
                "accessing option with incompatible template type", std::invalid_argument
            );
        }
    }

  private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, OptionValue> options_;
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
     * @brief Constructs an `Options` instance from option values.
     *
     * @param options A map of option keys to their corresponding option value.
     *
     * @throws std::invalid_argument If keys are not case-insensitive.
     */
    Options(std::unordered_map<std::string, OptionValue> options = {});


    /**
     * @brief Constructs an `Options` instance from option values as strings.
     *
     * @param options_as_strings A map of option keys to their string representations.
     *
     * @throws std::invalid_argument If keys are not case-insensitive.
     */
    Options(std::unordered_map<std::string, std::string> options_as_strings);

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
     *
     * @throws std::invalid_argument If the stored option type does not match T.
     * @throws std::bad_any_cast If `T` doesn't match the type of the option.
     *
     * @note Once a key has been accessed with a particular `T`, subsequent calls
     * to `get` on the same key must use the same `T`. Using a different `T` for
     * the same key will result in a `std::bad_any_cast`.
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
 *
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
 *
 * @throws std::invalid_argument If key_regex doesn't contain exactly one capture group.
 *
 * @see get_environment_variables(std::unordered_map<std::string, std::string>&,
 * std::string const&)
 */
std::unordered_map<std::string, std::string> get_environment_variables(
    std::string const& key_regex = "RAPIDSMPF_(.*)"
);

}  // namespace rapidsmpf::config
