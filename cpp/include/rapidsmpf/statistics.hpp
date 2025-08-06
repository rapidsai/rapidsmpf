/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {


/**
 * @class Statistics
 * @brief Track statistics across rapidsmpf operations.
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
     * @param mr Pointer to a memory resource used for memory profiling.
     *
     * @throws std::invalid_argument If `mr` is the nullptr.
     */
    Statistics(RmmResourceAdaptor* mr);

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
        : enabled_(o.enabled_), stats_{std::move(o.stats_)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The Statistics object to move from.
     * @return Reference to this updated instance.
     */
    Statistics& operator=(Statistics&& o) noexcept {
        enabled_ = o.enabled_;
        stats_ = std::move(o.stats_);
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
     * @brief Type alias for a statistics formatting function.
     *
     * The formatter is called with the output stream, update count, and accumulated
     * value.
     */
    using Formatter = std::function<void(std::ostream&, std::size_t, double)>;

    /**
     * @brief Default formatter for statistics output (implements `Formatter`).
     *
     * Prints the total value and, if count > 1, also prints the average.
     *
     * @param os Output stream to write the formatted result to.
     * @param count Number of updates to the statistic.
     * @param val Accumulated value.
     */
    static void FormatterDefault(std::ostream& os, std::size_t count, double val);

    /**
     * @brief Represents a single tracked statistic.
     */
    class Stat {
      public:
        /**
         * @brief Constructs a Stat with a specified formatter.
         *
         * @param formatter Function used to format this statistic when reporting.
         */
        Stat(Formatter formatter) : formatter_{std::move(formatter)} {}

        /**
         * @brief Equality operator for Stat objects.
         *
         * @param o Another Stat instance to compare with.
         * @return True if both the count and value are equal.
         */
        bool operator==(Stat const& o) const noexcept {
            return count_ == o.count() && value_ == o.value();
        }

        /**
         * @brief Adds a value to this statistic.
         *
         * Increments the update count and adds the given value.
         *
         * @param value The value to add.
         * @return The updated total value.
         */
        double add(double value) {
            ++count_;
            return value_ += value;
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
         * @brief Returns the formatter used by this statistic.
         *
         * @return A const reference to the formatter function.
         */
        [[nodiscard]] Formatter const& formatter() const noexcept {
            return formatter_;
        }

      private:
        std::size_t count_{0};
        double value_{0};
        Formatter formatter_;
    };

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
     * @param formatter Optional formatter to use for this statistic.
     * @return Updated total value.
     */
    double add_stat(
        std::string const& name,
        double value,
        Formatter const& formatter = FormatterDefault
    );

    /**
     * @brief Adds a byte count to the named statistic.
     *
     * Uses a formatter that formats values as human-readable byte sizes.
     *
     * @param name Name of the statistic.
     * @param nbytes Number of bytes to add.
     * @return The updated byte total.
     */
    std::size_t add_bytes_stat(std::string const& name, std::size_t nbytes);

    /**
     * @brief Adds a duration to the named statistic.
     *
     * Uses a formatter that formats values as time durations in seconds.
     *
     * @param name Name of the statistic.
     * @param seconds Duration in seconds to add.
     * @return The updated total duration.
     */
    Duration add_duration_stat(std::string const& name, Duration seconds);

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
    mutable std::mutex mutex_;
    bool enabled_;
    std::map<std::string, Stat> stats_;
    std::unordered_map<std::string, MemoryRecord> memory_records_;
    RmmResourceAdaptor* mr_;
};

/**
 * @brief Macro for automatic memory profiling of a code scope.
 *
 * This macro creates a scoped memory recorder that records memory usage statistics
 * upon entering and leaving a code block (if memory profiling is enabled).
 *
 * Example usage:
 * @code
 * void foo(Statistics& stats) {
 *     RAPIDSMPF_MEMORY_PROFILE(stats);
 *     RAPIDSMPF_MEMORY_PROFILE(stats, "custom_name");
 * }
 * @endcode
 *
 * @param stats A reference or pointer to a Statistics object.
 * @param funcname (optional) Custom function name string to use instead of __func__.
 */
#define RAPIDSMPF_MEMORY_PROFILE(...)                                       \
    RAPIDSMPF_OVERLOAD_BY_ARG_COUNT(                                        \
        __VA_ARGS__, RAPIDSMPF_MEMORY_PROFILE_2, RAPIDSMPF_MEMORY_PROFILE_1 \
    )                                                                       \
    (__VA_ARGS__)

// Version with default function name (__func__)
#define RAPIDSMPF_MEMORY_PROFILE_1(stats) RAPIDSMPF_MEMORY_PROFILE_2(stats, __func__)

// Version with custom function name
#define RAPIDSMPF_MEMORY_PROFILE_2(stats, funcname)                                \
    auto const RAPIDSMPF_CONCAT(_rapidsmpf_memory_recorder_, __LINE__) =           \
        ((rapidsmpf::detail::to_pointer(stats)                                     \
          && rapidsmpf::detail::to_pointer(stats)->is_memory_profiling_enabled())  \
             ? rapidsmpf::detail::to_pointer(stats)->create_memory_recorder(       \
                 std::string(__FILE__) + ":" + RAPIDSMPF_STRINGIFY(__LINE__) + "(" \
                 + std::string(funcname) + ")"                                     \
             )                                                                     \
             : rapidsmpf::Statistics::MemoryRecorder{})

}  // namespace rapidsmpf
