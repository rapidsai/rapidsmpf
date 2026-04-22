/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

class StreamOrderedTiming;

/**
 * @brief Tracks statistics across rapidsmpf operations.
 *
 * Two naming concepts are used throughout this class:
 *
 * - **Stat name**: identifies an individual `Stat` accumulator, as passed to
 *   `add_stat()`, `get_stat()`, `add_bytes_stat()`, and `add_duration_stat()`.
 *   Stats are pure numeric accumulators with no associated rendering
 *   information.
 *   Examples: `"spill-time"`, `"spill-bytes"`.
 *
 * - **Report entry name**: the label of a formatted line in `report()`,
 *   passed to `add_report_entry()`. An entry names one or more stats and
 *   a `Formatter` that selects how those stats are rendered. When the
 *   entry covers a single stat, the report entry name and stat name are
 *   typically identical.
 *   Example: `"spill"` (aggregating `"spill-bytes"` and `"spill-time"`).
 *
 * Formatters are a fixed, predefined set (see `Statistics::Formatter`).
 *
 * @code{.cpp}
 * Statistics stats;
 *
 * // Associate two stats with a predefined multi-stat formatter.
 * stats.add_report_entry(
 *     "copy-device-to-host",                // report entry name
 *     {"copy-device-to-host-bytes",
 *      "copy-device-to-host-time",
 *      "copy-device-to-host-stream-delay"},
 *     Statistics::Formatter::MemoryThroughput
 * );
 *
 * stats.add_bytes_stat("spill-bytes", 1024);    // helper: registers Bytes entry
 * stats.add_duration_stat("spill-time", 0.5s);  // helper: registers Duration entry
 *
 * auto s = stats.get_stat("spill-bytes");  // retrieve without formatter
 * std::cout << stats.report();
 * @endcode
 */
class Statistics {
  public:
    /**
     * @brief Identifies a predefined formatter used by `report()`.
     *
     * Each formatter consumes a fixed number of `Stat` entries and renders them
     * into a human-readable string.
     *
     * Available formatters (examples):
     *
     * - Default (1 stat):
     *   "123"
     *
     * - Bytes (1 stat):
     *   "1.2 GiB | avg 300 MiB"
     *
     * - Duration (1 stat):
     *   "2.5 ms | avg 600 us"
     *
     * - HitRate (1 stat):
     *   "42/100 (hits/lookups)"
     *
     * - MemoryThroughput (3 stats: bytes, time, stream-delay), where `stream-delay` is
     *   the wall-clock gap between CPU submission and GPU execution of the operation:
     *   "1.2 GiB | 2.5 ms | 480 GiB/s | avg-stream-delay 10 us"
     *
     * `_Count` is an internal sentinel — always keep it last.
     */
    enum class Formatter : std::uint8_t {
        Default = 0,
        Bytes,
        Duration,
        HitRate,
        MemoryThroughput,
        _Count,  ///< Sentinel; must remain last.
    };

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
     * @param mr The RMM resource adaptor used for memory profiling.
     * @param pinned_mr Optional pinned host memory resource for profiling; defaults to
     * `PinnedMemoryResource::Disabled`.
     */
    Statistics(
        RmmResourceAdaptor mr,
        std::shared_ptr<PinnedMemoryResource> pinned_mr = PinnedMemoryResource::Disabled
    );

    /**
     * @brief Construct from configuration options.
     *
     * @param mr The RMM resource adaptor used for memory profiling.
     * @param options Configuration options.
     * @param pinned_mr Optional pinned host memory resource for profiling; defaults to
     * `PinnedMemoryResource::Disabled`.
     *
     * @return A shared pointer to the constructed Statistics instance.
     */
    static std::shared_ptr<Statistics> from_options(
        RmmResourceAdaptor mr,
        config::Options options,
        std::shared_ptr<PinnedMemoryResource> pinned_mr = PinnedMemoryResource::Disabled
    );

    ~Statistics() noexcept;

    // `Statistics` is owned exclusively through `std::shared_ptr` (see `disabled()` and
    // `from_options()`).
    Statistics(Statistics const&) = delete;
    Statistics& operator=(Statistics const&) = delete;
    Statistics(Statistics&&) = delete;
    Statistics& operator=(Statistics&&) = delete;

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
     * @brief Checks if statistics tracking is enabled.
     *
     * @return True if statistics tracking is active, otherwise False.
     */
    bool enabled() const noexcept {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Enable statistics tracking for this instance.
     */
    void enable() noexcept {
        enabled_.store(true, std::memory_order_release);
    }

    /**
     * @brief Disable statistics tracking for this instance.
     */
    void disable() noexcept {
        enabled_.store(false, std::memory_order_release);
    }

    /**
     * @brief Generates a formatted report of all collected statistics.
     *
     * Every registered report entry always produces a line. If all the stats
     * it references have been recorded, the entry's `Formatter` renders
     * the values; otherwise the line reads "No data collected". Statistics
     * not covered by any report entry are shown with `Formatter::Default`
     * (raw numeric value, optionally annotated with the count). All entries
     * are sorted alphabetically.
     *
     * @note If any statistics are collected via stream-ordered timing (e.g. through
     * `record_copy()`), all relevant CUDA streams must be synchronized before calling
     * this method. Otherwise, some timing statistics may not yet have been recorded,
     * causing entries to read "No data collected" or imprecise statistics.
     *
     * @param header Header line prepended to the report.
     * @return Formatted statistics report.
     */
    std::string report(std::string const& header = "Statistics:") const;

    /**
     * @brief Writes a JSON representation of all collected statistics to a stream.
     *
     * Values are written as raw numbers (count, sum, max). Formatter
     * metadata is not emitted — use `report()` for the human-readable
     * rendering.
     *
     * @param os Output stream to write to.
     * @throws std::invalid_argument If any stat name or memory record name
     * contains characters that require JSON escaping (double quotes,
     * backslashes, or ASCII control characters 0x00–0x1F).
     */
    void write_json(std::ostream& os) const;

    /**
     * @brief Writes a JSON report of all collected statistics to a file.
     *
     * @param filepath Path to the output file. Created or overwritten.
     * @throws std::ios_base::failure If the file cannot be opened or writing fails.
     */
    void write_json(std::filesystem::path const& filepath) const;

    /**
     * @brief Creates a deep copy of this Statistics object.
     *
     * @note Memory records are not copied.
     *
     * @return A shared pointer to the new copy.
     */
    [[nodiscard]] std::shared_ptr<Statistics> copy() const;

    /**
     * @brief Serializes the stats and report entries to a binary byte vector.
     *
     * @note Memory records are not serialized.
     *
     * @return A vector of bytes containing the serialized statistics.
     */
    [[nodiscard]] std::vector<std::uint8_t> serialize() const;

    /**
     * @brief Deserializes a Statistics object from a binary byte vector.
     *
     * @note The resulting object has no memory records.
     *
     * @param data The serialized statistics data.
     * @return A shared pointer to the reconstructed Statistics object.
     * @throws std::invalid_argument If the data is malformed or truncated.
     */
    [[nodiscard]] static std::shared_ptr<Statistics> deserialize(
        std::span<std::uint8_t const> data
    );

    /**
     * @brief Merge a set of Statistics into a new instance.
     *
     * For each stat name present across the inputs, the result contains the
     * summed count, summed value, and the maximum of the recorded maxima.
     * The result's `enabled()` is true if any input is enabled. Memory
     * records are not merged.
     *
     * Report entries are unified by name. If multiple inputs contain the
     * same report-entry name, their `Formatter` and `stat_names` must match;
     * otherwise, this function throws `std::invalid_argument` to prevent
     * silent rendering inconsistencies (especially across
     * serialize/deserialize boundaries).
     *
     * @param stats Non-empty span of non-null `Statistics` instances to merge.
     * @return A new `Statistics` instance containing the merged data.
     *
     * @throws std::invalid_argument If @p stats is empty, contains a null
     * pointer, or if inputs disagree on the formatter or stat-name set for
     * a shared report entry.
     */
    [[nodiscard]] static std::shared_ptr<Statistics> merge(
        std::span<std::shared_ptr<Statistics> const> stats
    );

    /**
     * @brief Represents a single tracked statistic.
     *
     * @note Stat is not thread-safe. Thread safety is provided by the enclosing
     * Statistics object's mutex.
     */
    class Stat {
      public:
        /**
         * @brief Default-constructs a Stat.
         */
        Stat() = default;

        /**
         * @brief Constructs a Stat with explicit field values.
         *
         * @param count Number of updates.
         * @param value Total accumulated value.
         * @param max Maximum value seen.
         */
        Stat(std::size_t count, double value, double max);

        /**
         * @brief Three-way comparison operator.
         *
         * Performs memberwise comparison of all data members.
         *
         * @return The ordering result of the memberwise comparison.
         */
        auto operator<=>(Stat const&) const noexcept = default;

        /**
         * @brief Adds a value to this statistic.
         *
         * @param value The value to add.
         */
        void add(double value);

        /**
         * @brief Returns the number of updates applied to this statistic.
         *
         * @return The number of times `add()` was called.
         */
        [[nodiscard]] std::size_t count() const noexcept;

        /**
         * @brief Returns the total accumulated value.
         *
         * @return The sum of all values added.
         */
        [[nodiscard]] double value() const noexcept;

        /**
         * @brief Returns the maximum value seen across all `add()` calls.
         *
         * @return The maximum value added, or negative infinity if `add()` was never
         * called.
         */
        [[nodiscard]] double max() const noexcept;

        /**
         * @brief Returns the serialized size of this Stat in bytes.
         *
         * We size each field individually rather than using `sizeof(Stat)` to
         * avoid platform-dependent struct padding.
         *
         * @return The number of bytes needed to serialize this Stat.
         */
        [[nodiscard]] static constexpr std::size_t serialized_size() noexcept {
            return sizeof(std::uint64_t) + sizeof(double) + sizeof(double);
        }

        /**
         * @brief Serializes this Stat to a byte buffer.
         *
         * @param out Pointer to the output buffer. Must have at least
         * `serialized_size()` bytes available.
         * @return Pointer past the last byte written.
         */
        std::uint8_t* serialize(std::uint8_t* out) const;

        /**
         * @brief Deserializes a Stat from a byte buffer.
         *
         * @param data The input buffer. Must contain at least `serialized_size()`
         * bytes.
         * @return A pair of the deserialized Stat and the remaining unconsumed
         * bytes.
         * @throws std::invalid_argument If the data is truncated.
         */
        [[nodiscard]] static std::pair<Stat, std::span<std::uint8_t const>> deserialize(
            std::span<std::uint8_t const> data
        );

        /**
         * @brief Merges another Stat into this one, returning the combined result.
         *
         * Counts and values are summed; the maximum is taken.
         *
         * @param other The Stat to merge with.
         * @return A new Stat containing the merged result.
         */
        [[nodiscard]] Stat merge(Stat const& other) const;

      private:
        std::size_t count_{0};
        double value_{0};
        double max_{-std::numeric_limits<double>::infinity()};
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
     * Creates the statistic if it doesn't exist. Does not associate any
     * formatter with the stat — use `add_report_entry()` (or a helper like
     * `add_bytes_stat()`) for that.
     *
     * @param name Name of the statistic.
     * @param value Value to add.
     */
    void add_stat(std::string const& name, double value);

    /**
     * @brief Associate a formatter with one or more named statistics for
     * report rendering.
     *
     * First-wins: if a report entry is already registered under
     * @p report_entry_name, this call has no effect. The entry appears in
     * `report()` as a single line; if any stat it references is missing,
     * the line reads "No data collected".
     *
     * @param report_entry_name Report entry name.
     * @param stat_names Names of the stats this entry aggregates. Caller is
     * responsible for passing the number of stats the chosen @p formatter
     * expects; a mismatch surfaces as `std::out_of_range` when `report()`
     * renders the entry.
     * @param formatter Predefined formatter to render the entry with.
     */
    void add_report_entry(
        std::string const& report_entry_name,
        std::initializer_list<std::string_view> stat_names,
        Formatter formatter
    );

    // clang-format off
    /**
     * @copydoc add_report_entry(std::string const&,std::initializer_list<std::string_view>, Formatter)
     *
     * Overload for callers whose stat names come from a runtime container (e.g. the Python bindings).
     */
    // clang-format on
    void add_report_entry(
        std::string const& report_entry_name,
        std::vector<std::string> stat_names,
        Formatter formatter
    );

    /**
     * @brief Adds a byte count to the named statistic.
     *
     * Registers a `Formatter::Bytes` report entry named @p name if no
     * report entry already exists under that name, then adds @p nbytes to
     * the named statistic.
     *
     * @param name Name of the statistic.
     * @param nbytes Number of bytes to add.
     */
    void add_bytes_stat(std::string const& name, std::size_t nbytes);

    /**
     * @brief Adds a duration to the named statistic.
     *
     * Registers a `Formatter::Duration` report entry named @p name if no
     * report entry already exists under that name, then adds @p seconds to
     * the named statistic.
     *
     * @param name Name of the statistic.
     * @param seconds Duration in seconds to add.
     */
    void add_duration_stat(std::string const& name, Duration seconds);

    /**
     * @brief Record byte count and wall-clock duration for a memory copy operation.
     *
     * Records three statistics entries for `"copy-{src}-to-{dst}"`:
     *  - `"-bytes"`        — the number of bytes copied.
     *  - `"-time"`         — the copy duration, recorded in stream order.
     *  - `"-stream-delay"` — time between CPU submission and GPU execution of the copy,
     *                        recorded in stream order.
     *
     * All three entries are aggregated into a single combined report line under the name
     * `"copy-{src}-to-{dst}"`, showing total bytes, total time, bandwidth, and average
     * stream delay.
     *
     * @param src Source memory type.
     * @param dst Destination memory type.
     * @param nbytes Number of bytes copied.
     * @param timing A `StreamOrderedTiming` that should be started just before the copy
     * was enqueued on the stream. Its `stop_and_record()` is called here to enqueue the
     * stop callback.
     */
    void record_copy(
        MemoryType src, MemoryType dst, std::size_t nbytes, StreamOrderedTiming&& timing
    );

    /**
     * @brief Record size and wall-clock duration for a buffer allocation.
     *
     * Records three statistics entries for `"alloc-{memtype}"`:
     *  - `"-bytes"`        — the number of bytes allocated.
     *  - `"-time"`         — the allocation duration, recorded in stream order.
     *  - `"-stream-delay"` — time between CPU submission and GPU execution,
     *                        recorded in stream order.
     *
     * All three entries are aggregated into a single combined report line showing
     * total bytes, total time, throughput, and average stream delay.
     *
     * @param mem_type Memory type of the allocation.
     * @param nbytes Number of bytes allocated.
     * @param timing A `StreamOrderedTiming` constructed just before the allocation
     * was issued. Its `stop_and_record()` is called here.
     */
    void record_alloc(
        MemoryType mem_type, std::size_t nbytes, StreamOrderedTiming&& timing
    );

    /**
     * @brief Get the names of all statistics.
     *
     * @return A vector of all statistic names.
     */
    std::vector<std::string> list_stat_names() const;

    /**
     * @brief Clears all statistics.
     *
     * @note Memory profiling records and report entries are not cleared.
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
         * @param mr The RMM resource adaptor providing scoped memory statistics.
         * @param name Name of the scope.
         */
        MemoryRecorder(Statistics* stats, RmmResourceAdaptor mr, std::string name);

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
        std::optional<RmmResourceAdaptor> mr_;
        std::string name_;
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
     * @brief A report entry describing which stats to aggregate and how to render them.
     */
    struct ReportEntry {
        std::vector<std::string> stat_names;
        Formatter formatter;
    };

    mutable std::mutex mutex_;
    std::atomic<bool> enabled_;
    std::map<std::string, Stat> stats_;
    std::map<std::string, ReportEntry> report_entries_;
    std::unordered_map<std::string, MemoryRecord> memory_records_;
    std::optional<RmmResourceAdaptor> mr_;
    std::shared_ptr<PinnedMemoryResource>
        pinned_mr_;  ///< optional; not used by MemoryRecorder
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
