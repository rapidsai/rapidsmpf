/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>

/**
 * @brief Simple example demonstrating the use of CuptiMonitor
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
        rapidsmpf::CuptiMonitor monitor(true, 100);

        // Enable debug output for memory changes > 5MB
        monitor.set_debug_output(true, 5);

        std::cout << "Starting CUPTI monitoring...\n";
        monitor.start_monitoring();

        // Perform some CUDA memory operations to demonstrate monitoring
        const size_t num_allocations = 3;
        const size_t allocation_size = 64 * 1024 * 1024;  // 64MB each
        std::vector<float*> gpu_pointers;

        for (size_t i = 0; i < num_allocations; ++i) {
            float* d_data;

            std::cout << "Allocating " << allocation_size / (1024 * 1024)
                      << " MB on GPU...\n";
            cuda_err = cudaMalloc(&d_data, allocation_size);
            if (cuda_err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cuda_err)
                          << std::endl;
                break;
            }

            gpu_pointers.push_back(d_data);

            // Manually capture a memory sample
            monitor.capture_memory_sample();

            // Wait a bit to let periodic sampling work
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cout << "\nFreeing allocated memory...\n";

        // Free all allocated memory
        for (auto* ptr : gpu_pointers) {
            cudaFree(ptr);
        }

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

            std::cout << "Initial memory usage: "
                      << initial_sample.used_memory / (1024.0 * 1024.0) << " MB ("
                      << initial_sample.utilization_percent << "%)\n";
            std::cout << "Final memory usage: "
                      << final_sample.used_memory / (1024.0 * 1024.0) << " MB ("
                      << final_sample.utilization_percent << "%)\n";

            // Find peak memory usage
            size_t peak_used = 0;
            double peak_utilization = 0.0;
            for (const auto& sample : samples) {
                if (sample.used_memory > peak_used) {
                    peak_used = sample.used_memory;
                    peak_utilization = sample.utilization_percent;
                }
            }

            std::cout << "Peak memory usage: " << peak_used / (1024.0 * 1024.0) << " MB ("
                      << peak_utilization << "%)\n";
        }

        // Write results to CSV file
        std::string csv_filename = "cupti_monitor_example.csv";
        monitor.write_csv(csv_filename);
        std::cout << "Memory usage data written to " << csv_filename << "\n";

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
