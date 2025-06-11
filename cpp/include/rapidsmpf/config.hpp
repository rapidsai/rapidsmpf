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
 * and a string representation of the value.
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
     * @brief Constructs OptionValue from a string representation.
     *
     * @param value_as_string A string representation of the value.
     */
    OptionValue(std::string value_as_string)
        : value_as_string_{std::move(value_as_string)} {}

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
 * @brief Internal shared collection for the `Options` class.
 *
 * This struct is used internally by `Options` to share the options between
 * multiple instances of the Options class. This way, it is cheap to copy
 * Options and its values are only initialized once.
 */
struct SharedOptions {
    mutable std::mutex mutex;  ///< Shared mutex, must be use to guard `options`.
    std::unordered_map<std::string, OptionValue> options;  ///< Shared options.
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
 * to the shared options (`OptionsShared`).
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
     * @brief Inserts an option only if it is not already present.
     *
     * This method checks whether the given option key exists in the current
     * set of options. If it does not, the option is inserted in its string
     * representation.
     *
     * @param key The option key to insert. The key is trimmed and converted
     * to lower case before insertion.
     * @param option_as_string The string representation of the option value.
     *
     * @return `true` if the option was inserted; `false` if it was already present.
     */
    bool insert_if_absent(std::string const& key, std::string option_as_string);

    /**
     * @brief Inserts multiple options if they are not already present.
     *
     * This method attempts to insert each option key-value pair from the provided
     * map into the current set of options. Each insertion is performed only if the
     * key does not already exist in the options.
     *
     * @param options_as_strings A map of option keys to their string representations.
     * @return Number of newly inserted options (0 if none were added).
     */
    std::size_t insert_if_absent(
        std::unordered_map<std::string, std::string> options_as_strings
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
     *
     * @throws std::invalid_argument If the stored option type does not match T.
     * @throws std::bad_any_cast If `T` doesn't match the type of the option.
     *
     * @note Once a key has been accessed with a particular `T`, subsequent calls
     * to `get` on the same key must use the same `T`. Using a different `T` for
     * the same key will result in a `std::bad_any_cast`.
     */
    template <typename T>
    T const& get(const std::string& key, OptionFactory<T> factory) {
        auto& shared = *shared_;
        std::lock_guard<std::mutex> lock(shared.mutex);
        auto& option = shared.options[key];
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

    /**
     * @brief Retrieves all option values as strings.
     *
     * This method returns a map of all currently stored options where both the keys
     * and values are represented as strings.
     *
     * @return A map where each key is the option name and each value is the string
     * representation of the corresponding option's value.
     */
    [[nodiscard]] std::unordered_map<std::string, std::string> get_strings() const;

    /**
     * @brief Serializes the options into a binary buffer.
     *
     * An Options instance can only be serialized if no options have been accessed. This
     * is because serialization is based on the original string representations of the
     * options. Once an option has been accessed and parsed, its string value may no
     * longer accurately reflect its state, making serialization potentially inconsistent.
     *
     * The format is:
     * - [uint64_t count] — number of key-value pairs.
     * - [count * 2 * uint64_t] — offset pairs (key_offset, value_offset) for each entry.
     * - [raw bytes] — all key and value strings, contiguous and null-free.
     *
     * Offsets are absolute byte positions into the buffer.
     *
     * @return A byte vector representing the serialized options.
     *
     * @throws std::invalid_argument If any option has already been accessed.
     *
     * @note To ease Python/Cython compatibility, a std::vector<std::uint8_t> is returned
     * instead of std::vector<std::byte>.
     */
    [[nodiscard]] std::vector<std::uint8_t> serialize() const;

    /**
     * @brief Deserializes a binary buffer into an Options object.
     *
     * See Options::serialize() for the binary format.
     *
     * @param buffer The binary buffer produced by Options::serialize().
     * @return An Options object reconstructed from the buffer.
     *
     * @throws std::invalid_argument If the buffer is malformed or incomplete.
     * @throws std::out_of_range If offsets exceed buffer boundaries.
     */
    [[nodiscard]] static Options deserialize(std::vector<std::uint8_t> const& buffer);

  private:
    std::shared_ptr<detail::SharedOptions> shared_;
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
