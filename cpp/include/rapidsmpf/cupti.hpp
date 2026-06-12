/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef RAPIDSMPF_HAVE_CUPTI

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rapidsmpf {

/**
 * @brief Structure to hold memory usage data points.
 */
struct MemoryDataPoint {
    double timestamp;  ///< Time when sample was taken (seconds since epoch)
    std::size_t free_memory;  ///< Free GPU memory in bytes
    std::size_t total_memory;  ///< Total GPU memory in bytes
    std::size_t used_memory;  ///< Used GPU memory in bytes
};

/**
 * @brief Public wrapper for a CUPTI callback identifier.
 *
 * Keeps CUPTI types out of the public RapidsMPF API while preserving the
 * numeric callback identifier for callers that inspect callback counters.
 */
class CuptiCallbackId {
  public:
    using value_type = std::int32_t;

    constexpr CuptiCallbackId() noexcept = default;

    explicit constexpr CuptiCallbackId(value_type value) noexcept : value_(value) {}

    [[nodiscard]] constexpr value_type underlying() const noexcept {
        return value_;
    }

    friend constexpr bool operator==(
        CuptiCallbackId const&, CuptiCallbackId const&
    ) noexcept = default;

  private:
    value_type value_{};
};

}  // namespace rapidsmpf

template <>
struct std::hash<rapidsmpf::CuptiCallbackId> {
    std::size_t operator()(rapidsmpf::CuptiCallbackId callback_id) const noexcept {
        return std::hash<rapidsmpf::CuptiCallbackId::value_type>{}(
            callback_id.underlying()
        );
    }
};

namespace rapidsmpf {

/**
 * @brief CUDA memory monitoring using CUPTI (CUDA Profiling Tools Interface).
 *
 * This class provides memory monitoring capabilities for CUDA applications
 * by intercepting CUDA runtime and driver API calls related to memory
 * operations and kernel launches.
 */
class CuptiMonitor {
  public:
    /**
     * @brief Constructs a `CuptiMonitor` instance.
     *
     * @param enable_periodic_sampling Enable background thread for periodic memory
     * sampling.
     * @param sampling_interval_ms Interval between periodic samples in milliseconds.
     */
    explicit CuptiMonitor(
        bool enable_periodic_sampling = false,
        std::chrono::milliseconds sampling_interval_ms = std::chrono::milliseconds(100)
    );

    /**
     * @brief Destructor - automatically stops monitoring and cleans up CUPTI.
     */
    ~CuptiMonitor();

    // Delete copy constructor and assignment operator.
    CuptiMonitor(CuptiMonitor const&) = delete;
    CuptiMonitor& operator=(CuptiMonitor const&) = delete;

    // Delete move constructor and assignment operator.
    CuptiMonitor(CuptiMonitor&&) = delete;
    CuptiMonitor& operator=(CuptiMonitor&&) = delete;

    /**
     * @brief Start memory monitoring.
     *
     * Initializes CUPTI and begins intercepting CUDA API calls.
     *
     * @throws std::runtime_error if CUPTI initialization fails.
     */
    void start_monitoring();

    /**
     * @brief Stop memory monitoring.
     *
     * Stops CUPTI callbacks and periodic sampling if enabled.
     */
    void stop_monitoring();

    /**
     * @brief Check if monitoring is currently active.
     *
     * @return true if monitoring is active, false otherwise.
     */
    bool is_monitoring() const noexcept;

    /**
     * @brief Manually capture current memory usage.
     *
     * This can be called at any time to manually record a memory sample,
     * regardless of whether periodic sampling is enabled.
     */
    void capture_memory_sample();

    /**
     * @brief Get all collected memory samples.
     *
     * @return const reference to vector of memory data points.
     */
    std::vector<MemoryDataPoint> const& get_memory_samples() const noexcept;

    /**
     * @brief Clear all collected memory samples.
     */
    void clear_samples();

    /**
     * @brief Get the number of memory samples collected.
     *
     * @return number of samples.
     */
    std::size_t get_sample_count() const noexcept;

    /**
     * @brief Write memory samples to CSV file.
     *
     * @param filename Output CSV filename.
     * @throws std::runtime_error if file cannot be written.
     */
    void write_csv(std::string const& filename) const;

    /**
     * @brief Enable or disable debug output for significant memory changes.
     *
     * @param enabled if true, prints debug info when memory usage changes significantly.
     * @param threshold_mb threshold in MB for what constitutes a "significant" change.
     */
    void set_debug_output(bool enabled, std::size_t threshold_mb = 10);

    /**
     * @brief Get callback counters for all monitored CUPTI callbacks.
     *
     * Returns a map where keys are RapidsMPF callback ID wrappers and values are
     * the number of times each callback was triggered during monitoring.
     *
     * @return unordered_map from CuptiCallbackId to call count.
     */
    std::unordered_map<CuptiCallbackId, std::size_t> get_callback_counters() const;

    /**
     * @brief Clear all callback counters.
     *
     * Resets all callback counters to zero.
     */
    void clear_callback_counters();

    /**
     * @brief Get total number of callbacks triggered across all monitored callback IDs.
     *
     * @return total number of callbacks.
     */
    std::size_t get_total_callback_count() const;

    /**
     * @brief Get a human-readable summary of callback counters.
     *
     * Returns a formatted string showing callback names and their counts.
     *
     * @return string containing callback counter summary.
     */
    std::string get_callback_summary() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_CUPTI
