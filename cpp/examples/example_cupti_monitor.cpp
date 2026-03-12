/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <rmm/device_buffer.hpp>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>

/**
 * @brief Simple example demonstrating the use of CuptiMonitor.
 *
 * This example shows how to use RapidsMPF's CuptiMonitor to track
 * GPU memory usage during CUDA operations.
 */
int main() {
    std::cout << "CUPTI Memory Monitor Example\n";
    std::cout << "============================\n\n";

    try {
        // Initialize CUDA
        cudaError_t cuda_err = cudaSetDevice(0);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cuda_err)
                      << std::endl;
            return 1;
        }

        // Create a CuptiMonitor with periodic sampling enabled (every 100ms)
        rapidsmpf::CuptiMonitor monitor(true, std::chrono::milliseconds(100));

        // Enable debug output for memory changes > 5MB
        monitor.set_debug_output(true, 5);

        std::cout << "Starting CUPTI monitoring...\n";
        monitor.start_monitoring();

        // Perform some CUDA memory operations to demonstrate monitoring
        std::size_t const num_allocations = 3;
        std::size_t const allocation_size = 64 * 1024 * 1024;  // 64MB each
        std::vector<float*> gpu_pointers;

        // Use rmm::device_buffer to manage GPU memory allocations
        std::vector<rmm::device_buffer> device_buffers;

        for (std::size_t i = 0; i < num_allocations; ++i) {
            std::cout << "Allocating " << allocation_size / (1024 * 1024)
                      << " MB on GPU using rmm::device_buffer...\n";
            try {
                // Allocate device memory using rmm::device_buffer
                rmm::device_buffer buf(allocation_size, rmm::cuda_stream_default);
                device_buffers.push_back(std::move(buf));
            } catch (rmm::bad_alloc const& e) {
                std::cerr << "rmm::device_buffer allocation failed: " << e.what()
                          << std::endl;
                break;
            }

            // Manually capture a memory sample
            monitor.capture_memory_sample();

            // Wait a bit to let periodic sampling work
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cout << "\nReleasing allocated memory (handled by rmm::device_buffer "
                     "destructors)...\n";
        device_buffers.clear();

        // Capture final state
        monitor.capture_memory_sample();

        std::cout << "Stopping monitoring...\n";
        monitor.stop_monitoring();

        // Report results
        auto samples = monitor.get_memory_samples();
        std::cout << "\nMemory monitoring results:\n";
        std::cout << "Total samples collected: " << samples.size() << "\n";

        if (!samples.empty()) {
            auto initial_sample = samples.front();
            auto final_sample = samples.back();
            double initial_utilization = static_cast<double>(initial_sample.used_memory)
                                         / initial_sample.total_memory * 100.0;
            double final_utilization = static_cast<double>(final_sample.used_memory)
                                       / final_sample.total_memory * 100.0;

            std::cout << "Initial memory usage: "
                      << initial_sample.used_memory / (1024.0 * 1024.0) << " MB ("
                      << initial_utilization << "%)\n";
            std::cout << "Final memory usage: "
                      << final_sample.used_memory / (1024.0 * 1024.0) << " MB ("
                      << final_utilization << "%)\n";

            // Find peak memory usage
            std::size_t peak_used = 0;
            double peak_utilization = 0.0;
            for (const auto& sample : samples) {
                if (sample.used_memory > peak_used) {
                    peak_used = sample.used_memory;
                    peak_utilization = static_cast<double>(sample.used_memory)
                                       / sample.total_memory * 100.0;
                }
            }

            std::cout << "Peak memory usage: " << peak_used / (1024.0 * 1024.0) << " MB ("
                      << peak_utilization << "%)\n";
        }

        // Write results to CSV file
        std::string csv_filename = "cupti_monitor_example.csv";
        monitor.write_csv(csv_filename);
        std::cout << "Memory usage data written to " << csv_filename << "\n";

        // Show callback counter summary
        std::cout << "\n" << monitor.get_callback_summary() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nExample completed successfully!\n";
    return 0;
}

#else

int main() {
    std::cout
        << "CUPTI support is not enabled. Please build with -DBUILD_CUPTI_SUPPORT=ON\n";
    return 0;
}

#endif  // RAPIDSMPF_HAVE_CUPTI
