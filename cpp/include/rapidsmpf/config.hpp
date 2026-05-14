/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace config {

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
     * @brief Constructs OptionValue from a typed value.
     *
     * The value is stored directly and no string representation is provided.
     * Options constructed this way are considered initialized and make the
     * Options instance unserializable.
     *
     * @tparam T Type of the value to store.
     * @param value The value to store.
     */
    template <typename T>
    explicit OptionValue(T value)
        requires(!std::is_convertible_v<T, std::string_view>)
        : value_{std::make_any<T>(std::move(value))} {}

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
 * @brief Transparent hasher for `std::string`-keyed maps.
 *
 * Enables heterogeneous lookup so `find()` and friends accept
 * `std::string_view` (and `char const*`) without first constructing a
 * `std::string`, avoiding an allocation on every lookup.
 */
struct StringHash {
    /// @brief Opt-in marker that enables heterogeneous lookup on associative
    /// containers using this hasher (paired with `std::equal_to<>`).
    using is_transparent = void;

    /// @brief Hash a `std::string_view` (the canonical overload).
    /// @param sv The `std::string_view` to hash.
    /// @return The hash of the `std::string_view`.
    [[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }

    /// @brief Hash a `std::string` via its `string_view` view.
    /// @param s The `std::string` to hash.
    /// @return The hash of the `std::string`.
    [[nodiscard]] std::size_t operator()(std::string const& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }

    /// @brief Hash a null-terminated C string via its `string_view` view.
    /// @param s The null-terminated C string to hash.
    /// @return The hash of the null-terminated C string.
    [[nodiscard]] std::size_t operator()(char const* s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

/**
 * @brief Internal shared collection for the `Options` class.
 *
 * This struct is used internally by `Options` to share the options between
 * multiple instances of the Options class. This way, it is cheap to copy
 * Options and its values are only initialized once.
 */
struct SharedOptions {
    mutable std::mutex mutex;  ///< Shared mutex, must be use to guard `options`.
    /// @brief Shared options. Uses transparent hashing/equality so reads
    /// against descriptor keys avoid constructing a temporary `std::string`.
    std::unordered_map<std::string, OptionValue, StringHash, std::equal_to<>> options;
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
    bool insert_if_absent(std::string const& key, std::string_view option_as_string);

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
     * @brief Inserts an option only if it is not already present.
     *
     * This method stores a typed value directly, bypassing the string-based
     * representation used for lazy parsing. Once inserted, the option is
     * initialized, and subsequent calls to `get<T>()` for the same key must
     * use the same `T`.
     *
     * This method is only enabled for non string-like types. Values convertible
     * to `std::string_view` (for example `std::string`, `std::string_view`, or
     * string literals) are handled by the string-based overloads instead.
     *
     * Because no string representation is stored, inserting an option using
     * this method makes the Options instance unserializable. This is consistent
     * with the behavior of `get()`, as serialization relies exclusively on the
     * original string representations of options.
     *
     * @tparam T Type of the value to store.
     * @param key The option key to insert. The key is trimmed and converted to
     * lower case before insertion.
     * @param value The value to store.
     *
     * @return `true` if the option was inserted; `false` if it was already present.
     */
    template <typename T>
    bool insert_if_absent(std::string const& key, T value)
        requires(!std::is_convertible_v<T, std::string_view>)
    {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        auto [_, inserted] = shared_->options.try_emplace(
            to_lower(trim(key)), OptionValue{std::move(value)}
        );
        return inserted;
    }

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
    T const& get(std::string_view key, OptionFactory<T> factory) {
        auto& shared = *shared_;
        std::lock_guard<std::mutex> lock(shared.mutex);
        // Heterogeneous lookup avoids constructing a `std::string` when the
        // key is already present; only the first-insert path allocates.
        auto it = shared.options.find(key);
        if (it == shared.options.end()) {
            it = shared.options.try_emplace(std::string{key}).first;
        }
        auto& option = it->second;
        if (!option.get_value().has_value()) {
            option.set_value(std::make_any<T>(factory(option.get_value_as_string())));
        }
        try {
            return std::any_cast<T const&>(option.get_value());
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
     * Options that do not have a string representation, such as those inserted
     * using `insert_if_absent<T>()`, are included with an empty string value.
     *
     * @return A map where each key is the option name and each value is the string
     * representation of the corresponding option's value.
     */
    [[nodiscard]] std::unordered_map<std::string, std::string> get_strings() const;


    /**
     * @brief Serializes the options into a binary buffer.
     *
     * An Options instance can only be serialized if all options are still
     * represented exclusively by their original string values. Serialization
     * is based on these string representations and cannot reflect options that
     * have been accessed, parsed, or initialized with typed values.
     *
     * As a result, serialization is disallowed if any option has been accessed
     * via `get()` or inserted using the typed `insert_if_absent<T>()` method,
     * since their string values may no longer accurately reflect their state.
     *
     * The format (v1) is:
     * - [4 bytes MAGIC "RMPF"][1 byte version][1 byte flags][2 bytes reserved]
     * - [std::uint64_t count] — number of key-value pairs.
     * - [count * 2 * std::uint64_t] — offset pairs (key_offset, value_offset)
     *                                 for each entry.
     * - [raw bytes] — all key and value strings, contiguous and null-free.
     *
     * Offsets are absolute byte positions into the buffer.
     *
     * Serialization limits:
     * - Maximum options: 65,536 entries
     * - Maximum key size: 4 KiB
     * - Maximum value size: 1 MiB
     * - Maximum total buffer size: 64 MiB
     *
     * @return A byte vector representing the serialized options.
     *
     * @throws std::invalid_argument If any option has already been accessed or inserted
     * as a typed value.
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

}  // namespace config

/**
 * @brief Compile-time descriptor for a single configuration option.
 *
 * Couples an option's lookup key with its default value so the two cannot
 * drift apart. Instances are `inline constexpr` and live in the module
 * sub-namespaces below (e.g. `rapidsmpf::statistics`,
 * `rapidsmpf::buffer_resource`); consult those for the canonical list of
 * options understood by the `from_options` factories.
 *
 * Each descriptor's variable name is suffixed with `Option` to keep it
 * distinct from same-named runtime entities in its module (for example,
 * `rapidsmpf::ucxx::ProgressModeOption` vs the `enum class ProgressMode`).
 *
 * Both `key` and `default_val` are stored as `std::string_view`. Options are
 * always parsed from their string representation at runtime, so the default
 * is expressed as a string and fed through the same factory the call site
 * uses for user-supplied values. Descriptors must be initialized from
 * string literals so that `key.data()` and `default_val.data()` yield
 * null-terminated `char const*` pointers, which are consumed directly by
 * the Cython bindings.
 */
struct OptionDescriptor {
    std::string_view key;  ///< Lookup key passed to `Options::get`.
    std::string_view default_val;  ///< String form of the value used when unset.
};

/// @brief Options for `rapidsmpf::Statistics::from_options`.
namespace statistics {
/// @brief Whether statistics tracking is enabled.
inline constexpr OptionDescriptor EnabledOption{
    .key = "statistics",
    .default_val = "false",
};
}  // namespace statistics

/// @brief Options for `rapidsmpf::PinnedMemoryResource::from_options`.
namespace pinned_memory {
/// @brief Whether pinned host memory is enabled.
inline constexpr OptionDescriptor EnabledOption{
    .key = "pinned_memory",
    .default_val = "false",
};

/// @brief Initial pinned-pool size, applied as
/// `get_host_memory_per_gpu() * InitialPoolSizeOption`.
inline constexpr OptionDescriptor InitialPoolSizeOption{
    .key = "pinned_initial_pool_size",
    .default_val = "0%",
};

/// @brief Maximum pinned-pool size, applied as
/// `get_host_memory_per_gpu() * MaxPoolSizeOption`.
inline constexpr OptionDescriptor MaxPoolSizeOption{
    .key = "pinned_max_pool_size",
    .default_val = "80%",
};
}  // namespace pinned_memory

/// @brief Options for `rapidsmpf::BufferResource::from_options` and helpers.
namespace buffer_resource {
/// @brief Device-memory spill limit (nbytes string or percent of total).
inline constexpr OptionDescriptor SpillDeviceLimitOption{
    .key = "spill_device_limit",
    .default_val = "80%",
};

/// @brief Periodic spill-check interval (duration string or
/// disabled-sentinel).
inline constexpr OptionDescriptor PeriodicSpillCheckOption{
    .key = "periodic_spill_check",
    .default_val = "1ms",
};

/// @brief CUDA stream-pool size used by the buffer resource.
inline constexpr OptionDescriptor NumStreamsOption{
    .key = "num_streams",
    .default_val = "16",
};
}  // namespace buffer_resource

/// @brief Options for the streaming subsystem.
namespace streaming {
/// @brief Number of threads in the streaming coroutine pool.
inline constexpr OptionDescriptor NumStreamingThreadsOption{
    .key = "num_streaming_threads",
    .default_val = "1",
};

/// @brief Per-attempt timeout for streaming memory reservations.
inline constexpr OptionDescriptor MemoryReserveTimeoutOption{
    .key = "memory_reserve_timeout",
    .default_val = "100 ms",
};

/// @brief Whether streaming memory reservations may overbook by default.
/// Used by `reserve_memory` when the caller does not pass an explicit
/// `AllowOverbooking` policy.
inline constexpr OptionDescriptor AllowOverbookingByDefaultOption{
    .key = "allow_overbooking_by_default",
    .default_val = "true",
};

}  // namespace streaming

/// @brief Options consumed by `rapidsmpf::Communicator::Logger`.
namespace communicator {
/// @brief Logger verbosity level (string form, one of
/// `Logger::LOG_LEVEL_NAMES`).
inline constexpr OptionDescriptor LogOption{
    .key = "log",
    .default_val = "WARN",
};
}  // namespace communicator

/// @brief Options consumed by `rapidsmpf::ucxx::init`.
namespace ucxx {
/// @brief UCXX worker progress mode; one of `"blocking"`, `"polling"`,
/// `"thread-blocking"`, `"thread-polling"`.
inline constexpr OptionDescriptor ProgressModeOption{
    .key = "ucxx_progress_mode",
    .default_val = "thread-blocking",
};
}  // namespace ucxx

}  // namespace rapidsmpf
