/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <compare>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {


/**
 * @brief Track statistics across rapidsmpf operations.
 *
 * Two distinct naming concepts are used throughout this class:
 *
 * - **Stat name**: the key used to identify an individual `Stat` accumulator,
 *   as passed to `add_stat()`, `get_stat()`, `add_bytes_stat()`, and
 *   `add_duration_stat()`. A stat name may be used exclusively — accumulated
 *   via `add_stat()` and retrieved via `get_stat()` — without ever registering
 *   a formatter or appearing in the report.
 *   Examples: `"spill-time"`, `"spill-bytes"`.
 *
 * - **Report entry name**: the label of a formatted line in the output of
 *   `report()`, passed to `register_formatter()`. A single report entry may
 *   aggregate one or more stats. When using the single-stat overload of
 *   `register_formatter()`, the report entry name and the stat name are the
 *   same string.
 *   Examples: `"spill"` (aggregating both `"spill-time"` and `"spill-bytes"`).
 *
 * Example:
 * @code{.cpp}
 * Statistics stats;
 *
 * // Register a report entry that aggregates two stats under one label.
 * stats.register_formatter(
 *     "spill",                              // report entry name
 *     {"spill-bytes", "spill-time"},        // stat names
 *     [](std::ostream& os, auto const& s) {
 *         os << format_nbytes(s[0].value()) << " in " << format_duration(s[1].value());
 *     }
 * );
 *
 * // Accumulate values using stat names.
 * stats.add_bytes_stat("spill-bytes", 1024);
 * stats.add_duration_stat("spill-time", 0.5);
 *
 * // Retrieve a stat directly by stat name (no formatter needed).
 * auto s = stats.get_stat("spill-bytes");
 *
 * // Produce the report (uses report entry names as labels).
 * std::cout << stats.report();
 * @endcode
 */
class Statistics {
  public:
    /**
     * @brief Constructs a Statistics object without memory profiling.
     *
     * @param enabled If true, enables tracking of statistics. If false, all operations
     * are no-ops.
     */
    Statistics(bool enabled = true);

    /**
     * @brief Constructs a Statistics object with memory profiling enabled.
     *
     * Automatically enables both statistics and memory profiling.
     *
     * @param mr Pointer to a memory resource used for memory profiling. Must remain valid
     * for the lifetime of the returned object.
     *
     * @throws std::invalid_argument If `mr` is the nullptr.
     */
    Statistics(RmmResourceAdaptor* mr);

    /**
     * @brief Construct from configuration options.
     *
     * @param mr Pointer to a memory resource used for memory profiling. Must remain valid
     * for the lifetime of the returned object.
     * @param options Configuration options.
     *
     * @return A shared pointer to the constructed Statistics instance.
     */
    static std::shared_ptr<Statistics> from_options(
        RmmResourceAdaptor* mr, config::Options options
    );

    ~Statistics() noexcept = default;
    Statistics(Statistics const&) = delete;
    Statistics& operator=(Statistics const&) = delete;

    /**
     * @brief Returns a shared pointer to a disabled (no-op) Statistics instance.
     *
     * Useful when you need to pass a Statistics reference but do not want to
     * collect any data.
     *
     * @return A shared pointer to a Statistics instance with tracking disabled.
     */
    static std::shared_ptr<Statistics> disabled();

    /**
     * @brief Move constructor.
     *
     * @param o The Statistics object to move from.
     */
    Statistics(Statistics&& o) noexcept
        : enabled_(o.enabled_),
          stats_{std::move(o.stats_)},
          formatters_{std::move(o.formatters_)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The Statistics object to move from.
     * @return Reference to this updated instance.
     */
    Statistics& operator=(Statistics&& o) noexcept {
        enabled_ = o.enabled_;
        stats_ = std::move(o.stats_);
        formatters_ = std::move(o.formatters_);
        return *this;
    }

    /**
     * @brief Checks if statistics tracking is enabled.
     *
     * @return True if statistics tracking is active, otherwise False.
     */
    bool enabled() const noexcept {
        return enabled_;
    }

    /**
     * @brief Generates a formatted report of all collected statistics.
     *
     * @param header An optional header to prepend to the report.
     * @return A string containing the formatted statistics.
     */
    std::string report(std::string const& header = "Statistics:") const;

    /**
     * @brief Represents a single tracked statistic.
     */
    class Stat {
      public:
        /**
         * @brief Default-constructs a Stat.
         */
        Stat() = default;

        /**
         * @brief Three-way comparison operator.
         *
         * Performs lexicographical comparison of all data members.
         */
        auto operator<=>(Stat const&) const noexcept = default;

        /**
         * @brief Adds a value to this statistic.
         *
         * @param value The value to add.
         */
        void add(double value) {
            ++count_;
            value_ += value;
            max_ = std::max(max_, value);
        }

        /**
         * @brief Returns the number of updates applied to this statistic.
         *
         * @return The number of times `add()` was called.
         */
        [[nodiscard]] std::size_t count() const noexcept {
            return count_;
        }

        /**
         * @brief Returns the total accumulated value.
         *
         * @return The sum of all values added.
         */
        [[nodiscard]] double value() const noexcept {
            return value_;
        }

        /**
         * @brief Returns the maximum value seen across all `add()` calls.
         *
         * @return The maximum value added, or negative infinity if `add()` was never
         * called.
         */
        [[nodiscard]] double max() const noexcept {
            return max_;
        }

      private:
        std::size_t count_{0};
        double value_{0};
        double max_{-std::numeric_limits<double>::infinity()};
    };

    /**
     * @brief Type alias for a statistics formatting function.
     *
     * The formatter receives all the named stats it declared interest in as a vector.
     */
    using Formatter = std::function<void(std::ostream&, std::vector<Stat> const&)>;

    /**
     * @brief Retrieves a statistic by name.
     *
     * @param name Name of the statistic.
     * @return The requested statistic.
     */
    Stat get_stat(std::string const& name) const;

    /**
     * @brief Adds a numeric value to the named statistic.
     *
     * Creates the statistic if it doesn't exist.
     *
     * @param name Name of the statistic.
     * @param value Value to add.
     */
    void add_stat(std::string const& name, double value);

    /**
     * @brief Check whether a report entry name already has a formatter registered.
     *
     * Intended as a cheap pre-check before constructing arguments to
     * `register_formatter()`.
     *
     * @note The result may be outdated by the time it is acted upon. This method
     * should only be used as an optimization hint to avoid unnecessary work, never
     * for correctness decisions. Once this method returns `true` for a given name it
     * will never return `false` again, because formatters cannot be unregistered.
     *
     * @param name Report entry name to look up.
     * @return True if a formatter is registered under @p name, otherwise false.
     */
    bool exist_report_entry_name(std::string const& name) const;

    /**
     * @brief Register a formatter for a single named statistic.
     *
     * If a formatter is already registered under @p name, this call has no effect.
     * The formatter is only invoked during `report()` if the named statistic has
     * been recorded.
     *
     * @param name Report entry name (also used as the stat name to collect).
     * @param formatter Function used to format this statistic when reporting.
     */
    void register_formatter(std::string const& name, Formatter formatter);

    /**
     * @brief Register a formatter that takes multiple named statistics.
     *
     * If a formatter is already registered under @p report_entry_name, this call has
     * no effect. The formatter is only invoked during `report()` if all stats listed
     * in @p stat_names have been recorded; if any are missing the entry is silently
     * omitted.
     *
     * @param report_entry_name Report entry name.
     * @param stat_names Names of the stats to collect and pass to the formatter.
     * @param formatter Function called with all collected stats during reporting.
     */
    void register_formatter(
        std::string const& report_entry_name,
        std::vector<std::string> const& stat_names,
        Formatter formatter
    );

    /**
     * @brief Adds a byte count to the named statistic.
     *
     * Registers a formatter that formats values as human-readable byte sizes if no
     * formatter is already registered for @p name, then adds @p nbytes to the
     * named statistic.
     *
     * @param name Name of the statistic.
     * @param nbytes Number of bytes to add.
     */
    void add_bytes_stat(std::string const& name, std::size_t nbytes);

    /**
     * @brief Adds a duration to the named statistic.
     *
     * Registers a formatter that formats values as time durations in seconds if no
     * formatter is already registered for @p name, then adds @p seconds to the
     * named statistic.
     *
     * @param name Name of the statistic.
     * @param seconds Duration in seconds to add.
     */
    void add_duration_stat(std::string const& name, Duration seconds);

    /**
     * @brief Record byte count for a memory copy operation.
     *
     * Records one statistics entry:
     *  - `"copy-{src}-to-{dst}"` — the number of bytes copied.
     *
     * @param src Source memory type.
     * @param dst Destination memory type.
     * @param nbytes Number of bytes copied.
     */
    void record_copy(MemoryType src, MemoryType dst, std::size_t nbytes);

    /**
     * @brief Get the names of all statistics.
     *
     * @return A vector of all statistic names.
     */
    std::vector<std::string> list_stat_names() const;

    /**
     * @brief Clears all statistics.
     *
     * @note Memory profiling records and registered formatters are not cleared.
     */
    void clear();

    /**
     * @brief Checks whether memory profiling is enabled.
     *
     * @return True if memory profiling is active, otherwise False.
     */
    bool is_memory_profiling_enabled() const;

    /**
     * @brief Holds memory profiling information for a named scope.
     */
    struct MemoryRecord {
        ScopedMemoryRecord scoped;  ///< Scoped memory stats.
        std::int64_t global_peak{0};  ///< Peak global memory usage during the scope.
        std::uint64_t num_calls{0};  ///< Number of times the scope was invoked.
    };

    /**
     * @brief RAII-style object for scoped memory usage tracking.
     *
     * Automatically tracks memory usage between construction and destruction.
     */
    class MemoryRecorder {
      public:
        /**
         * @brief Constructs a no-op MemoryRecorder (disabled state).
         */
        MemoryRecorder() = default;

        /**
         * @brief Constructs an active MemoryRecorder.
         *
         * @param stats Pointer to Statistics object that will store the result.
         * @param mr Memory resource that provides the scoped memory statistics.
         * @param name Name of the scope.
         */
        MemoryRecorder(Statistics* stats, RmmResourceAdaptor* mr, std::string name);

        /**
         * @brief Destructor.
         *
         * Captures memory counters and stores them in the Statistics object.
         */
        ~MemoryRecorder();

        /// Deleted copy and move constructors/assignments
        MemoryRecorder(MemoryRecorder const&) = delete;
        MemoryRecorder& operator=(MemoryRecorder const&) = delete;
        MemoryRecorder(MemoryRecorder&&) = delete;
        MemoryRecorder& operator=(MemoryRecorder&&) = delete;

      private:
        Statistics* stats_{nullptr};
        RmmResourceAdaptor* mr_{nullptr};
        std::string name_;
        ScopedMemoryRecord main_record_;
    };

    /**
     * @brief Creates a scoped memory recorder for the given name.
     *
     * If memory profiling is not enabled, returns a no-op recorder.
     *
     * @param name Name of the scope.
     * @return A MemoryRecorder instance.
     */
    MemoryRecorder create_memory_recorder(std::string name);

    /**
     * @brief Retrieves all memory profiling records stored by this instance.
     *
     * @return A reference to a map from record name to memory usage data.
     */
    std::unordered_map<std::string, MemoryRecord> const& get_memory_records() const;

  private:
    /**
     * @brief Associates a display name with a formatter and the stats it aggregates.
     */
    struct FormatterEntry {
        std::vector<std::string> stat_names;  ///< Stats to collect and pass to fn.
        Formatter fn;
    };

    mutable std::mutex mutex_;
    bool enabled_;
    std::map<std::string, Stat> stats_;
    std::map<std::string, FormatterEntry> formatters_;
    std::unordered_map<std::string, MemoryRecord> memory_records_;
    RmmResourceAdaptor* mr_;
};

/**
 * @brief Macro for automatic memory profiling of a code scope.
 *
 * This macro creates a scoped memory recorder that records memory usage statistics
 * upon entering and leaving a code block (if memory profiling is enabled).
 *
 * Usage:
 * - `RAPIDSMPF_MEMORY_PROFILE(stats)` - Uses __func__ as the function name
 * - `RAPIDSMPF_MEMORY_PROFILE(stats, "custom_name")` - Uses custom_name as the function
 * name
 *
 * Example usage:
 * @code
 * void foo(Statistics& stats) {
 *     RAPIDSMPF_MEMORY_PROFILE(stats);
 *     RAPIDSMPF_MEMORY_PROFILE(stats, "custom_name");
 * }
 * @endcode
 *
 * The first argument is a reference or pointer to a Statistics object.
 * The second argument (optional) is a custom function name string to use instead of
 * __func__.
 */
#define RAPIDSMPF_MEMORY_PROFILE(...)                                       \
    RAPIDSMPF_OVERLOAD_BY_ARG_COUNT(                                        \
        __VA_ARGS__, RAPIDSMPF_MEMORY_PROFILE_2, RAPIDSMPF_MEMORY_PROFILE_1 \
    )                                                                       \
    (__VA_ARGS__)

// Version with default function name (__func__)
#define RAPIDSMPF_MEMORY_PROFILE_1(stats) RAPIDSMPF_MEMORY_PROFILE_2(stats, __func__)

// Version with custom function name
#define RAPIDSMPF_MEMORY_PROFILE_2(stats, funcname)                                      \
    auto&& RAPIDSMPF_CONCAT(_rapidsmpf_stats_, __LINE__) = (stats);                      \
    auto const RAPIDSMPF_CONCAT(_rapidsmpf_memory_recorder_, __LINE__) =                 \
        ((rapidsmpf::detail::to_pointer(RAPIDSMPF_CONCAT(_rapidsmpf_stats_, __LINE__))   \
          && rapidsmpf::detail::to_pointer(                                              \
                 RAPIDSMPF_CONCAT(_rapidsmpf_stats_, __LINE__)                           \
          ) -> is_memory_profiling_enabled())                                            \
             ? rapidsmpf::detail::to_pointer(                                            \
                   RAPIDSMPF_CONCAT(_rapidsmpf_stats_, __LINE__)                         \
               )                                                                         \
                   ->create_memory_recorder(                                             \
                       std::string(__FILE__) + ":" + RAPIDSMPF_STRINGIFY(__LINE__) + "(" \
                       + std::string(funcname) + ")"                                     \
                   )                                                                     \
             : rapidsmpf::Statistics::MemoryRecorder{})

}  // namespace rapidsmpf
