/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

#ifdef RAPIDSMPF_HAVE_CUPTI

class CuptiMonitorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize CUDA context
        cudaError_t cuda_err = cudaSetDevice(0);
        ASSERT_EQ(cuda_err, cudaSuccess)
            << "Failed to set CUDA device: " << cudaGetErrorString(cuda_err);
    }

    void TearDown() override {
        // Clean up any remaining GPU allocations
        cudaDeviceSynchronize();
    }

    // Helper function to allocate and free GPU memory
    void perform_gpu_operations(std::size_t size_bytes, int num_operations = 1) {
        std::vector<rmm::device_buffer> buffers;

        // Allocate memory using rmm::device_buffer
        for (int i = 0; i < num_operations; ++i) {
            try {
                buffers.emplace_back(size_bytes, rmm::cuda_stream_default);
            } catch (const rmm::bad_alloc& e) {
                FAIL() << "rmm::device_buffer allocation failed: " << e.what();
            }
        }
        // Buffers will be automatically freed when going out of scope
    }
};

TEST_F(CuptiMonitorTest, MemoryDataPointStructure) {
    // Test that MemoryDataPoint structure is properly defined
    rapidsmpf::MemoryDataPoint point;
    point.timestamp = 1.0;
    point.free_memory = 1000;
    point.total_memory = 2000;
    point.used_memory = 1000;

    EXPECT_EQ(point.timestamp, 1.0);
    EXPECT_EQ(point.free_memory, 1000);
    EXPECT_EQ(point.total_memory, 2000);
    EXPECT_EQ(point.used_memory, 1000);
}

TEST_F(CuptiMonitorTest, BasicConstruction) {
    // Test default construction
    CuptiMonitor monitor1;
    EXPECT_FALSE(monitor1.is_monitoring());
    EXPECT_EQ(monitor1.get_sample_count(), 0);

    // Test construction with parameters
    CuptiMonitor monitor2(
        true, std::chrono::milliseconds(50)
    );  // periodic sampling every 50ms
    EXPECT_FALSE(monitor2.is_monitoring());
    EXPECT_EQ(monitor2.get_sample_count(), 0);
}

TEST_F(CuptiMonitorTest, StartStopMonitoring) {
    CuptiMonitor monitor;

    // Initially not monitoring
    EXPECT_FALSE(monitor.is_monitoring());

    // Start monitoring
    ASSERT_NO_THROW(monitor.start_monitoring());
    EXPECT_TRUE(monitor.is_monitoring());

    // Should have captured initial state
    EXPECT_GT(monitor.get_sample_count(), 0);

    // Stop monitoring
    monitor.stop_monitoring();
    EXPECT_FALSE(monitor.is_monitoring());

    // Should have captured final state
    auto final_count = monitor.get_sample_count();
    EXPECT_GT(final_count, 1);  // At least initial + final

    // Stopping again should be safe
    ASSERT_NO_THROW(monitor.stop_monitoring());
    EXPECT_FALSE(monitor.is_monitoring());
}

TEST_F(CuptiMonitorTest, DoubleStartMonitoring) {
    CuptiMonitor monitor;

    // Start monitoring twice - should be safe
    ASSERT_NO_THROW(monitor.start_monitoring());
    EXPECT_TRUE(monitor.is_monitoring());
    auto first_count = monitor.get_sample_count();

    ASSERT_NO_THROW(monitor.start_monitoring());
    EXPECT_TRUE(monitor.is_monitoring());

    // Should not have added extra samples
    EXPECT_EQ(monitor.get_sample_count(), first_count);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, ManualCapture) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    auto initial_count = monitor.get_sample_count();

    // Manual capture should add a sample
    monitor.capture_memory_sample();
    EXPECT_EQ(monitor.get_sample_count(), initial_count + 1);

    // Multiple manual captures
    monitor.capture_memory_sample();
    monitor.capture_memory_sample();
    EXPECT_EQ(monitor.get_sample_count(), initial_count + 3);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, ManualCaptureWithoutMonitoring) {
    CuptiMonitor monitor;

    // Manual capture without monitoring should be safe but no-op
    ASSERT_NO_THROW(monitor.capture_memory_sample());
    EXPECT_EQ(monitor.get_sample_count(), 0);
}

TEST_F(CuptiMonitorTest, MemoryOperationsDetection) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    auto initial_count = monitor.get_sample_count();

    // Perform GPU memory operations - should trigger CUPTI callbacks
    perform_gpu_operations(1_MiB, 3);

    auto final_count = monitor.get_sample_count();
    // Should have captured memory allocations/deallocations
    EXPECT_GT(final_count, initial_count);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, MemoryDataPoints) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    // Perform some operations
    perform_gpu_operations(2_MiB);

    monitor.stop_monitoring();

    auto samples = monitor.get_memory_samples();
    ASSERT_GT(samples.size(), 0);

    // Check data point structure
    for (const auto& sample : samples) {
        EXPECT_GT(sample.timestamp, 0);
        EXPECT_GT(sample.total_memory, 0);
        EXPECT_LE(sample.free_memory, sample.total_memory);
        EXPECT_EQ(sample.used_memory, sample.total_memory - sample.free_memory);
    }

    // Timestamps should be in order
    for (std::size_t i = 1; i < samples.size(); ++i) {
        EXPECT_GE(samples[i].timestamp, samples[i - 1].timestamp);
    }
}

TEST_F(CuptiMonitorTest, ClearSamples) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    perform_gpu_operations(1_MiB);

    EXPECT_GT(monitor.get_sample_count(), 0);
    auto samples_before = monitor.get_memory_samples();
    EXPECT_GT(samples_before.size(), 0);

    // Clear samples
    monitor.clear_samples();
    EXPECT_EQ(monitor.get_sample_count(), 0);

    auto samples_after = monitor.get_memory_samples();
    EXPECT_EQ(samples_after.size(), 0);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, PeriodicSampling) {
    // Monitor with periodic sampling every 50ms
    CuptiMonitor monitor(true, std::chrono::milliseconds(50));
    monitor.start_monitoring();

    auto initial_count = monitor.get_sample_count();

    // Wait for periodic samples to be collected
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto final_count = monitor.get_sample_count();

    // Should have collected periodic samples
    EXPECT_GT(final_count, initial_count);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, NoPeriodicSampling) {
    // Monitor without periodic sampling
    CuptiMonitor monitor(false, std::chrono::milliseconds(50));
    monitor.start_monitoring();

    auto initial_count = monitor.get_sample_count();

    // Wait - should not collect periodic samples
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto final_count = monitor.get_sample_count();

    // Should only have initial sample (no periodic sampling)
    EXPECT_EQ(final_count, initial_count);

    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, CSVExport) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    perform_gpu_operations(1_MiB, 2);

    monitor.stop_monitoring();

    auto rank = GlobalEnvironment->comm_->rank();
    std::string filename = "test_cupti_output_" + std::to_string(rank) + ".csv";

    // Write CSV
    ASSERT_NO_THROW(monitor.write_csv(filename));

    // Verify file exists and has content
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Should have header + data lines
    ASSERT_GT(lines.size(), 1);

    // Check header
    EXPECT_THAT(lines[0], ::testing::HasSubstr("timestamp"));
    EXPECT_THAT(lines[0], ::testing::HasSubstr("free_memory_bytes"));
    EXPECT_THAT(lines[0], ::testing::HasSubstr("total_memory_bytes"));
    EXPECT_THAT(lines[0], ::testing::HasSubstr("used_memory_bytes"));

    // Check data lines have correct number of columns
    for (std::size_t i = 1; i < lines.size(); ++i) {
        auto comma_count = std::count(lines[i].begin(), lines[i].end(), ',');
        EXPECT_EQ(comma_count, 3);  // 4 columns = 3 commas
    }

    // Clean up
    std::remove(filename.c_str());
}

TEST_F(CuptiMonitorTest, CSVExportInvalidPath) {
    CuptiMonitor monitor;
    monitor.start_monitoring();
    monitor.stop_monitoring();

    // Try to write to invalid path
    EXPECT_THROW(monitor.write_csv("/invalid/path/file.csv"), std::runtime_error);
}

TEST_F(CuptiMonitorTest, DebugOutput) {
    CuptiMonitor monitor;

    // Test setting debug output
    ASSERT_NO_THROW(monitor.set_debug_output(true, 5));  // 5MB threshold
    ASSERT_NO_THROW(monitor.set_debug_output(false, 10));  // Disable

    // These calls should be safe regardless of monitoring state
    monitor.start_monitoring();
    ASSERT_NO_THROW(monitor.set_debug_output(true, 1));  // 1MB threshold
    monitor.stop_monitoring();
}

TEST_F(CuptiMonitorTest, ThreadSafety) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    std::vector<std::thread> threads;
    const int num_threads = 4;

    // Multiple threads performing operations simultaneously
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &monitor, i]() {
            // Each thread does some GPU operations and manual captures
            perform_gpu_operations(1_MiB);
            monitor.capture_memory_sample();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            monitor.capture_memory_sample();
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    monitor.stop_monitoring();

    // Should have collected samples from all threads
    EXPECT_GT(monitor.get_sample_count(), num_threads);

    // All samples should be valid
    auto samples = monitor.get_memory_samples();
    for (const auto& sample : samples) {
        EXPECT_GT(sample.total_memory, 0);
        EXPECT_LE(sample.free_memory, sample.total_memory);
    }
}

TEST_F(CuptiMonitorTest, DestructorCleanup) {
    {
        CuptiMonitor monitor;
        monitor.start_monitoring();
        perform_gpu_operations(1_MiB);

        // Monitor should be destroyed here and automatically stop monitoring
    }

    // Should be able to create a new monitor after destruction
    CuptiMonitor monitor2;
    ASSERT_NO_THROW(monitor2.start_monitoring());
    monitor2.stop_monitoring();
}

TEST_F(CuptiMonitorTest, LargeNumberOfSamples) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    // Generate many samples
    for (int i = 0; i < 100; ++i) {
        monitor.capture_memory_sample();
    }

    EXPECT_EQ(monitor.get_sample_count(), 101);  // 100 manual + 1 initial

    auto samples = monitor.get_memory_samples();
    EXPECT_EQ(samples.size(), 101);

    monitor.stop_monitoring();
    EXPECT_EQ(monitor.get_sample_count(), 102);  // +1 final
}

TEST_F(CuptiMonitorTest, CallbackCounters) {
    CuptiMonitor monitor;

    // Initially no callbacks
    EXPECT_EQ(monitor.get_total_callback_count(), 0);
    auto counters = monitor.get_callback_counters();
    EXPECT_TRUE(counters.empty());

    monitor.start_monitoring();

    // Perform GPU operations that should trigger callbacks
    perform_gpu_operations(1_MiB, 2);

    monitor.stop_monitoring();

    // Should have recorded some callbacks
    EXPECT_GT(monitor.get_total_callback_count(), 0);

    counters = monitor.get_callback_counters();
    EXPECT_FALSE(counters.empty());

    // Verify that callback summary doesn't crash and contains expected content
    std::string summary = monitor.get_callback_summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_THAT(summary, ::testing::HasSubstr("CUPTI Callback Counter Summary"));
    EXPECT_THAT(summary, ::testing::HasSubstr("Total"));
}

TEST_F(CuptiMonitorTest, CallbackCountersClear) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    perform_gpu_operations(1_MiB);

    monitor.stop_monitoring();

    // Should have some callbacks recorded
    EXPECT_GT(monitor.get_total_callback_count(), 0);

    // Clear counters
    monitor.clear_callback_counters();

    // Should be empty now
    EXPECT_EQ(monitor.get_total_callback_count(), 0);
    auto counters = monitor.get_callback_counters();
    EXPECT_TRUE(counters.empty());

    // Summary should indicate no callbacks
    std::string summary = monitor.get_callback_summary();
    EXPECT_THAT(summary, ::testing::HasSubstr("No callbacks recorded yet"));
}

TEST_F(CuptiMonitorTest, CallbackCountersAccumulate) {
    CuptiMonitor monitor;
    monitor.start_monitoring();

    // First batch of operations
    perform_gpu_operations(1_MiB, 1);
    auto first_count = monitor.get_total_callback_count();

    // Second batch of operations
    perform_gpu_operations(1_MiB, 1);
    auto second_count = monitor.get_total_callback_count();

    // Should have accumulated more callbacks
    EXPECT_GT(second_count, first_count);

    monitor.stop_monitoring();
}

#else

// Tests when CUPTI is not available
TEST(CuptiMonitorTest, CuptiNotAvailable) {
    // This test runs when CUPTI support is not compiled in
    GTEST_SKIP() << "CUPTI support not enabled. Build with -DBUILD_CUPTI_SUPPORT=ON to "
                    "enable tests.";
}

#endif  // RAPIDSMPF_HAVE_CUPTI
