#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Simple example demonstrating the use of CuptiMonitor.

This example shows how to use RapidsMPF's CuptiMonitor to track
GPU memory usage during CUDA operations.
"""

from __future__ import annotations

import rmm

try:
    from rapidsmpf.cupti import CuptiMonitor

    CUPTI_AVAILABLE = True
except ImportError:
    CUPTI_AVAILABLE = False


def main() -> None:
    """Main example function demonstrating CUPTI memory monitoring."""
    print("CUPTI Memory Monitor Example")
    print("============================\n")

    if not CUPTI_AVAILABLE:
        print(
            "CUPTI support is not available. Please ensure CUPTI support is compiled in."
        )
        return

    try:
        # Create a CuptiMonitor with periodic sampling enabled (every 100ms)
        monitor = CuptiMonitor(enable_periodic_sampling=True, sampling_interval_ms=100)

        # Enable debug output for memory changes > 5MB
        monitor.set_debug_output(enabled=True, threshold_mb=5)

        print("Starting CUPTI monitoring...")
        monitor.start_monitoring()

        # Perform some GPU memory operations to demonstrate monitoring
        num_allocations = 3
        allocation_size = 64 * 1024 * 1024  # 64MB each
        device_buffers: list[rmm.DeviceBuffer] = []

        for _ in range(num_allocations):
            print(
                f"Allocating {allocation_size // (1024 * 1024)} MB on GPU using rmm.DeviceBuffer..."
            )
            try:
                # Allocate device memory using rmm.DeviceBuffer
                buf = rmm.DeviceBuffer(size=allocation_size)
                device_buffers.append(buf)
            except Exception as e:
                print(f"rmm.DeviceBuffer allocation failed: {e}")
                break

            # Manually capture a memory sample
            monitor.capture_memory_sample()

        print("\nReleasing allocated memory (handled by rmm.DeviceBuffer cleanup)...")
        device_buffers.clear()

        print("Stopping monitoring...")
        monitor.stop_monitoring()

        # Report results
        samples = monitor.get_memory_samples()
        print("\nMemory monitoring results:")
        print(f"Total samples collected: {len(samples)}")

        if samples:
            initial_sample = samples[0]
            final_sample = samples[-1]
            initial_utilization = (
                initial_sample.used_memory / initial_sample.total_memory
            ) * 100.0
            final_utilization = (
                final_sample.used_memory / final_sample.total_memory
            ) * 100.0

            print(
                f"Initial memory usage: {initial_sample.used_memory / (1024.0 * 1024.0):.2f} MB "
                f"({initial_utilization:.1f}%)"
            )
            print(
                f"Final memory usage: {final_sample.used_memory / (1024.0 * 1024.0):.2f} MB "
                f"({final_utilization:.1f}%)"
            )

            # Find peak memory usage
            peak_used = 0
            peak_utilization = 0.0
            for sample in samples:
                if sample.used_memory > peak_used:
                    peak_used = sample.used_memory
                    peak_utilization = (
                        sample.used_memory / sample.total_memory
                    ) * 100.0

            print(
                f"Peak memory usage: {peak_used / (1024.0 * 1024.0):.2f} MB "
                f"({peak_utilization:.1f}%)"
            )

        # Write results to CSV file
        csv_filename = "cupti_monitor_example.csv"
        monitor.write_csv(csv_filename)
        print(f"Memory usage data written to {csv_filename}")

        # Show callback counter summary
        print(f"\n{monitor.get_callback_summary()}")

    except Exception as e:
        print(f"Error: {e}")
        return

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
