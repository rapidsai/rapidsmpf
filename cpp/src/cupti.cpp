/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef RAPIDSMPF_HAVE_CUPTI

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <sstream>
#include <stdexcept>

#include <rapidsmpf/cupti.hpp>

namespace rapidsmpf {

namespace {

// List of CUDA Runtime API callbacks we want to monitor. We don't monitor any runtime
// callbacks by default because they are redundant with the driver API callbacks.
constexpr std::array<CUpti_CallbackId, 0> MONITORED_RUNTIME_CALLBACKS{{}};

// List of CUDA Driver API callbacks we want to monitor
constexpr std::array<CUpti_CallbackId, 17> MONITORED_DRIVER_CALLBACKS{{
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost,
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemFree,
    CUPTI_DRIVER_TRACE_CBID_cu64MemFree,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost,
    CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz,
}};

// Helper function to check if a callback ID is in our monitored list
template <std::size_t N>
bool is_monitored_callback(
    CUpti_CallbackId cbid, std::array<CUpti_CallbackId, N> const& monitored_list
) {
    return std::find(monitored_list.begin(), monitored_list.end(), cbid)
           != monitored_list.end();
}

}  // namespace

CuptiMonitor::CuptiMonitor(
    bool enable_periodic_sampling, std::chrono::milliseconds sampling_interval_ms
)
    : enable_periodic_sampling_(enable_periodic_sampling),
      monitoring_active_(false),
      sampling_interval_ms_(sampling_interval_ms),
      debug_threshold_bytes_(10 * 1024 * 1024) {}  // 10MB default

CuptiMonitor::~CuptiMonitor() {
    stop_monitoring();
}

void CuptiMonitor::start_monitoring() {
    if (monitoring_active_.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);

        CUptiResult cupti_err = subscribe();
        if (cupti_err != CUPTI_SUCCESS) {
            throw std::runtime_error(
                "Failed to initialize CUPTI: " + std::to_string(cupti_err)
            );
        }
    }

    monitoring_active_.store(true);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Capture initial memory state
        capture_memory_usage_impl();

        if (enable_periodic_sampling_) {
            sampling_thread_ = std::thread(&CuptiMonitor::periodic_memory_sampling, this);
        }
    }
}

void CuptiMonitor::stop_monitoring() {
    if (!monitoring_active_.load()) {
        return;
    }

    monitoring_active_.store(false);

    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Capture final memory state
        capture_memory_usage_impl();
    }

    unsubscribe();
}

bool CuptiMonitor::is_monitoring() const noexcept {
    return monitoring_active_.load();
}

void CuptiMonitor::capture_memory_sample() {
    if (monitoring_active_.load()) {
        std::lock_guard<std::mutex> lock(mutex_);
        capture_memory_usage_impl();
    }
}

std::vector<MemoryDataPoint> const& CuptiMonitor::get_memory_samples() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_samples_;
}

void CuptiMonitor::clear_samples() {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_samples_.clear();
}

std::size_t CuptiMonitor::get_sample_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_samples_.size();
}

void CuptiMonitor::write_csv(std::string const& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename + " for writing");
    }

    // Write CSV header
    file << "timestamp,free_memory_bytes,total_memory_bytes,used_memory_bytes\n";

    // Write data points
    for (auto const& point : memory_samples_) {
        file << std::fixed << std::setprecision(6) << point.timestamp << ","
             << point.free_memory << "," << point.total_memory << "," << point.used_memory
             << "\n";
    }
}

void CuptiMonitor::set_debug_output(bool enabled, std::size_t threshold_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    debug_output_enabled_ = enabled;
    debug_threshold_bytes_ = threshold_mb * 1024 * 1024;  // Convert MB to bytes
}

std::unordered_map<CUpti_CallbackId, std::size_t>
CuptiMonitor::get_callback_counters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return callback_counters_;
}

void CuptiMonitor::clear_callback_counters() {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_counters_.clear();
}

std::size_t CuptiMonitor::get_total_callback_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t total = 0;
    for (auto const& [cbid, count] : callback_counters_) {
        total += count;
    }
    return total;
}

// Helper function to get human-readable name for callback ID
std::string get_callback_name(CUpti_CallbackId cbid) {
    switch (cbid) {
    // Runtime API callbacks
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
        return "cudaMalloc";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020:
        return "cudaMallocPitch";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020:
        return "cudaMallocArray";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
        return "cudaMallocHost";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020:
        return "cudaMalloc3D";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020:
        return "cudaMalloc3DArray";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000:
        return "cudaMallocMipmappedArray";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
        return "cudaMallocManaged";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020:
        return "cudaMallocAsync";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020:
        return "cudaMallocAsync_ptsz";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_v11020:
        return "cudaMallocFromPoolAsync";
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_ptsz_v11020:
        return "cudaMallocFromPoolAsync_ptsz";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
        return "cudaFree";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020:
        return "cudaFreeArray";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
        return "cudaFreeHost";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000:
        return "cudaFreeMipmappedArray";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020:
        return "cudaFreeAsync";
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020:
        return "cudaFreeAsync_ptsz";

    // Driver API callbacks
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc:
        return "cuMemAlloc";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch:
        return "cuMemAllocPitch";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost:
        return "cuMemAllocHost";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
        return "cuMemAlloc_v2";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2:
        return "cuMemAllocPitch_v2";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
        return "cuMemAllocHost_v2";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged:
        return "cuMemAllocManaged";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync:
        return "cuMemAllocAsync";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz:
        return "cuMemAllocAsync_ptsz";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync:
        return "cuMemAllocFromPoolAsync";
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz:
        return "cuMemAllocFromPoolAsync_ptsz";
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree:
        return "cuMemFree";
    case CUPTI_DRIVER_TRACE_CBID_cu64MemFree:
        return "cu64MemFree";
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost:
        return "cuMemFreeHost";
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
        return "cuMemFree_v2";
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync:
        return "cuMemFreeAsync";
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz:
        return "cuMemFreeAsync_ptsz";

    default:
        return "Unknown_" + std::to_string(cbid);
    }
}

std::string CuptiMonitor::get_callback_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (callback_counters_.empty()) {
        return "No callbacks recorded yet.\n";
    }

    std::ostringstream ss;
    ss << "CUPTI Callback Counter Summary:\n";
    ss << "==============================\n";

    std::size_t total = 0;

    // Sort callbacks by count (descending) for better readability
    std::vector<std::pair<CUpti_CallbackId, std::size_t>> sorted_callbacks(
        callback_counters_.begin(), callback_counters_.end()
    );
    std::ranges::sort(sorted_callbacks, std::greater{}, [](auto const& v) {
        return v.second;
    });

    for (auto const& [cbid, count] : sorted_callbacks) {
        ss << std::setw(35) << std::left << get_callback_name(cbid) << ": "
           << std::setw(10) << std::right << count << " calls\n";
        total += count;
    }

    ss << "==============================\n";
    ss << std::setw(35) << std::left << "Total"
       << ": " << std::setw(10) << std::right << total << " calls\n";

    return ss.str();
}

void CuptiMonitor::capture_memory_usage_impl() {
    std::size_t free_mem, total_mem;
    cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);

    if (cuda_status == cudaSuccess) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        double timestamp = std::chrono::duration<double>(duration).count();

        std::size_t used_mem = total_mem - free_mem;
        double utilization = static_cast<double>(used_mem) / total_mem * 100.0;

        MemoryDataPoint point = {
            .timestamp = timestamp,
            .free_memory = free_mem,
            .total_memory = total_mem,
            .used_memory = used_mem
        };

        memory_samples_.push_back(point);

        // Debug output for significant memory changes
        if (debug_output_enabled_) {
            if (std::abs(
                    static_cast<long>(used_mem)
                    - static_cast<long>(last_used_mem_for_debug_)
                )
                > static_cast<long>(debug_threshold_bytes_))
            {
                std::printf(
                    "Memory change: %.1f MB -> %.1f MB (%.2f%%)\n",
                    last_used_mem_for_debug_ / (1024.0 * 1024.0),
                    used_mem / (1024.0 * 1024.0),
                    utilization
                );
                last_used_mem_for_debug_ = used_mem;
            }
        }
    }
}

void CuptiMonitor::periodic_memory_sampling() {
    while (monitoring_active_.load()) {
        capture_memory_sample();
        std::this_thread::sleep_for(sampling_interval_ms_);
    }
}

CUptiResult CuptiMonitor::subscribe() {
    CUptiResult cupti_err;

    cupti_err = cuptiSubscribe(&cupti_subscriber_, callback_wrapper, this);
    if (cupti_err != CUPTI_SUCCESS) {
        return cupti_err;
    }

    // Enable runtime API callbacks using our centralized list
    for (auto const& cbid : MONITORED_RUNTIME_CALLBACKS) {
        cupti_err =
            cuptiEnableCallback(1, cupti_subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, cbid);
        if (cupti_err != CUPTI_SUCCESS) {
            return cupti_err;
        }
    }

    // Enable driver API callbacks using our centralized list
    for (auto const& cbid : MONITORED_DRIVER_CALLBACKS) {
        cupti_err =
            cuptiEnableCallback(1, cupti_subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid);
        if (cupti_err != CUPTI_SUCCESS) {
            return cupti_err;
        }
    }

    return cupti_err;
}

void CuptiMonitor::unsubscribe() {
    cuptiUnsubscribe(cupti_subscriber_);
}

void CUPTIAPI CuptiMonitor::callback_wrapper(
    void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const* cbdata
) {
    auto* monitor = static_cast<CuptiMonitor*>(userdata);
    monitor->callback(domain, cbid, cbdata);
}

void CuptiMonitor::callback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const* cbdata
) {
    if (!monitoring_active_.load())
        return;

    auto cbInfo = static_cast<CUpti_CallbackData const*>(cbdata);

    // Check if this callback is one we're monitoring
    bool should_monitor = false;
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        should_monitor = is_monitored_callback(cbid, MONITORED_RUNTIME_CALLBACKS);
    } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        should_monitor = is_monitored_callback(cbid, MONITORED_DRIVER_CALLBACKS);
    }

    if (should_monitor && cbInfo->callbackSite == CUPTI_API_EXIT) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_counters_[cbid]++;
        }

        capture_memory_sample();
    }
}

}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_CUPTI
