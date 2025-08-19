/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef RAPIDSMPF_HAVE_CUPTI

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <unistd.h>

#include <rapidsmpf/cupti.hpp>

namespace rapidsmpf {

CuptiMonitor::CuptiMonitor(
    bool enable_periodic_sampling, std::size_t sampling_interval_ms
)
    : enable_periodic_sampling_(enable_periodic_sampling),
      sampling_interval_ms_(sampling_interval_ms),
      monitoring_active_(false),
      debug_output_enabled_(false),
      debug_threshold_bytes_(10 * 1024 * 1024),  // 10MB default
      last_used_mem_for_debug_(0) {}

CuptiMonitor::~CuptiMonitor() {
    if (monitoring_active_) {
        stop_monitoring();
    }
}

void CuptiMonitor::start_monitoring() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (monitoring_active_) {
        return;  // Already monitoring
    }

    // Initialize CUPTI
    CUptiResult cupti_err = init_cupti();
    if (cupti_err != CUPTI_SUCCESS) {
        throw std::runtime_error(
            "Failed to initialize CUPTI: " + std::to_string(cupti_err)
        );
    }

    monitoring_active_ = true;

    // Capture initial memory state
    capture_memory_usage_impl();

    // Start periodic sampling thread if enabled
    if (enable_periodic_sampling_) {
        sampling_thread_ = std::thread(&CuptiMonitor::periodic_memory_sampling, this);
    }
}

void CuptiMonitor::stop_monitoring() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!monitoring_active_) {
        return;  // Already stopped
    }

    monitoring_active_ = false;

    // Stop periodic sampling thread
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }

    // Capture final memory state
    capture_memory_usage_impl();

    // Cleanup CUPTI
    cleanup_cupti();
}

bool CuptiMonitor::is_monitoring() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return monitoring_active_;
}

void CuptiMonitor::capture_memory_sample() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (monitoring_active_) {
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
    for (const auto& point : memory_samples_) {
        file << std::fixed << std::setprecision(6) << point.timestamp << ","
             << point.free_memory << "," << point.total_memory << "," << point.used_memory
             << "\n";
    }

    file.close();
}

void CuptiMonitor::set_debug_output(bool enabled, std::size_t threshold_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    debug_output_enabled_ = enabled;
    debug_threshold_bytes_ = threshold_mb * 1024 * 1024;  // Convert MB to bytes
}

void CuptiMonitor::capture_memory_usage_from_callback() {
    if (monitoring_active_) {
        std::lock_guard<std::mutex> lock(mutex_);
        capture_memory_usage_impl();
    }
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

        MemoryDataPoint point = {timestamp, free_mem, total_mem, used_mem};

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
    while (monitoring_active_) {
        capture_memory_sample();
        usleep(sampling_interval_ms_ * 1000);  // Convert ms to microseconds
    }
}

CUptiResult CuptiMonitor::init_cupti() {
    CUptiResult cupti_err;

    // Subscribe to runtime API callbacks
    cupti_err = cuptiSubscribe(&cupti_subscriber_, cupti_callback_wrapper, this);
    if (cupti_err != CUPTI_SUCCESS) {
        return cupti_err;
    }

    // Enable runtime API callbacks
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
    );

    // Enable driver API callbacks
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2
    );
    cupti_err = cuptiEnableCallback(
        1,
        cupti_subscriber_,
        CUPTI_CB_DOMAIN_DRIVER_API,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
    );

    return cupti_err;
}

void CuptiMonitor::cleanup_cupti() {
    cuptiUnsubscribe(cupti_subscriber_);
}

// Static wrapper function for CUPTI callback
void CUPTIAPI CuptiMonitor::cupti_callback_wrapper(
    void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata
) {
    auto* monitor = static_cast<CuptiMonitor*>(userdata);
    monitor->cupti_callback(domain, cbid, cbdata);
}

// Instance method for CUPTI callback
void CuptiMonitor::cupti_callback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata
) {
    if (!monitoring_active_)
        return;

    const CUpti_CallbackData* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                capture_memory_usage_from_callback();
            }
            break;
        }
    } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                capture_memory_usage_from_callback();
            }
            break;
        }
    }
}

}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_CUPTI
