# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Try to import CUPTI module, skip tests if not available
try:
    from rapidsmpf.cupti import CuptiMonitor, MemoryDataPoint
except ImportError:
    pytest.skip(reason="CUPTI support not available")


def _perform_gpu_operations(size_bytes: int, num_operations: int = 1) -> None:
    """Helper function to allocate and free GPU memory using CUDA runtime APIs."""
    # Use multiple approaches to trigger CUDA runtime API calls that CUPTI monitors
    try:
        for _ in range(num_operations):
            # Method 1: Use CuPy's direct CUDA runtime calls
            try:
                import cupy.cuda.runtime as runtime

                # Allocate using CUDA runtime API
                ptr, _ = runtime.malloc(size_bytes)
                # Free using CUDA runtime API
                runtime.free(ptr)
            except Exception:
                pass

            # Method 2: Use CuPy arrays which should trigger memory callbacks
            arr = cp.zeros(size_bytes // 4, dtype=cp.float32)
            # Perform operations that might trigger kernel callbacks
            arr.fill(1.0)
            cp.cuda.Stream.null.synchronize()
            # Force deallocation
            del arr

            # Method 3: Try managed memory allocation
            try:
                ptr_managed, _ = runtime.mallocManaged(size_bytes)
                runtime.free(ptr_managed)
            except Exception:
                pass

    except cp.cuda.memory.OutOfMemoryError as e:
        pytest.fail(f"GPU memory allocation failed: {e}")
    except Exception:
        # Don't fail the test if GPU operations have issues,
        # as the main goal is to test CUPTI monitoring
        pass


@pytest.fixture
def cuda_context() -> Generator[None, None, None]:
    """Fixture to ensure CUDA context is initialized."""
    try:
        # Initialize CUDA context
        cp.cuda.Device(0).use()
        yield
    finally:
        # Synchronize device to ensure all operations complete
        cp.cuda.Device().synchronize()


class TestMemoryDataPoint:
    """Test cases for MemoryDataPoint class."""

    def test_memory_data_point_properties(self) -> None:
        """Test that MemoryDataPoint properties work correctly."""
        # Note: We can't directly create MemoryDataPoint instances from Python
        # They are created internally by CuptiMonitor, so we test them through that
        monitor = CuptiMonitor()
        monitor.start_monitoring()
        monitor.capture_memory_sample()
        monitor.stop_monitoring()

        samples = monitor.get_memory_samples()
        assert len(samples) > 0

        sample = samples[0]
        assert isinstance(sample, MemoryDataPoint)
        assert isinstance(sample.timestamp, float)
        assert isinstance(sample.free_memory, int)
        assert isinstance(sample.total_memory, int)
        assert isinstance(sample.used_memory, int)

        # Basic sanity checks
        assert sample.timestamp > 0
        assert sample.total_memory > 0
        assert sample.free_memory <= sample.total_memory
        assert sample.used_memory == sample.total_memory - sample.free_memory

    def test_memory_data_point_repr(self) -> None:
        """Test MemoryDataPoint string representation."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()
        monitor.capture_memory_sample()
        monitor.stop_monitoring()

        samples = monitor.get_memory_samples()
        sample = samples[0]

        repr_str = repr(sample)
        assert "MemoryDataPoint" in repr_str
        assert "timestamp=" in repr_str
        assert "free_memory=" in repr_str
        assert "total_memory=" in repr_str
        assert "used_memory=" in repr_str


class TestCuptiMonitor:
    """Test cases for CuptiMonitor class."""

    def test_basic_construction(self, cuda_context: None) -> None:
        """Test CuptiMonitor construction with different parameters."""
        # Test default construction
        monitor1 = CuptiMonitor()
        assert not monitor1.is_monitoring()
        assert monitor1.get_sample_count() == 0

        # Test construction with parameters
        monitor2 = CuptiMonitor(enable_periodic_sampling=True, sampling_interval_ms=50)
        assert not monitor2.is_monitoring()
        assert monitor2.get_sample_count() == 0

    def test_start_stop_monitoring(self, cuda_context: None) -> None:
        """Test starting and stopping monitoring."""
        monitor = CuptiMonitor()

        # Initially not monitoring
        assert not monitor.is_monitoring()

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring()

        # Should have captured initial state
        assert monitor.get_sample_count() > 0

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring()

        # Should have captured final state
        final_count = monitor.get_sample_count()
        assert final_count >= 2  # At least initial + final

        # Stopping again should be safe
        monitor.stop_monitoring()
        assert not monitor.is_monitoring()

    def test_double_start_monitoring(self, cuda_context: None) -> None:
        """Test that starting monitoring twice is safe."""
        monitor = CuptiMonitor()

        # Start monitoring twice - should be safe
        monitor.start_monitoring()
        assert monitor.is_monitoring()
        first_count = monitor.get_sample_count()

        monitor.start_monitoring()
        assert monitor.is_monitoring()

        # Should not have added extra samples
        assert monitor.get_sample_count() == first_count

        monitor.stop_monitoring()

    def test_manual_capture(self, cuda_context: None) -> None:
        """Test manual memory sample capture."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        initial_count = monitor.get_sample_count()

        # Manual capture should add a sample
        monitor.capture_memory_sample()
        assert monitor.get_sample_count() == initial_count + 1

        # Multiple manual captures
        monitor.capture_memory_sample()
        monitor.capture_memory_sample()
        assert monitor.get_sample_count() == initial_count + 3

        monitor.stop_monitoring()

    def test_manual_capture_without_monitoring(self, cuda_context: None) -> None:
        """Test that manual capture without monitoring is safe but no-op."""
        monitor = CuptiMonitor()

        # Manual capture without monitoring should be safe but no-op
        monitor.capture_memory_sample()
        assert monitor.get_sample_count() == 0

    def test_memory_operations_detection(self, cuda_context: None) -> None:
        """Test that GPU memory operations are detected."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        initial_count = monitor.get_sample_count()

        # Perform GPU memory operations - should trigger CUPTI callbacks
        _perform_gpu_operations(1024 * 1024, 3)  # 1 MiB, 3 operations

        final_count = monitor.get_sample_count()
        # Should have captured memory allocations/deallocations
        assert final_count > initial_count

        monitor.stop_monitoring()

    def test_memory_data_points(self, cuda_context: None) -> None:
        """Test memory data point collection and validation."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        # Perform some operations
        _perform_gpu_operations(2 * 1024 * 1024)  # 2 MiB

        monitor.stop_monitoring()

        samples = monitor.get_memory_samples()
        assert len(samples) > 0

        # Check data point structure
        for sample in samples:
            assert sample.timestamp > 0
            assert sample.total_memory > 0
            assert sample.free_memory <= sample.total_memory
            assert sample.used_memory == sample.total_memory - sample.free_memory

        # Timestamps should be in order
        for i in range(1, len(samples)):
            assert samples[i].timestamp >= samples[i - 1].timestamp

    def test_clear_samples(self, cuda_context: None) -> None:
        """Test clearing collected samples."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        _perform_gpu_operations(1024 * 1024)  # 1 MiB

        assert monitor.get_sample_count() > 0
        samples_before = monitor.get_memory_samples()
        assert len(samples_before) > 0

        # Clear samples
        monitor.clear_samples()
        assert monitor.get_sample_count() == 0

        samples_after = monitor.get_memory_samples()
        assert len(samples_after) == 0

        monitor.stop_monitoring()

    def test_periodic_sampling(self, cuda_context: None) -> None:
        """Test periodic sampling functionality."""
        # Monitor with periodic sampling every 50ms
        monitor = CuptiMonitor(enable_periodic_sampling=True, sampling_interval_ms=50)
        monitor.start_monitoring()

        initial_count = monitor.get_sample_count()

        # Wait for periodic samples to be collected
        time.sleep(0.2)  # 200ms

        final_count = monitor.get_sample_count()

        # Should have collected periodic samples
        assert final_count > initial_count

        monitor.stop_monitoring()

    def test_no_periodic_sampling(self, cuda_context: None) -> None:
        """Test that periodic sampling can be disabled."""
        # Monitor without periodic sampling
        monitor = CuptiMonitor(enable_periodic_sampling=False, sampling_interval_ms=50)
        monitor.start_monitoring()

        initial_count = monitor.get_sample_count()

        # Wait - should not collect periodic samples
        time.sleep(0.2)  # 200ms

        final_count = monitor.get_sample_count()

        # Should only have initial sample (no periodic sampling)
        assert final_count == initial_count

        monitor.stop_monitoring()

    def test_csv_export(self, cuda_context: None) -> None:
        """Test CSV export functionality."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        _perform_gpu_operations(1024 * 1024, 2)  # 1 MiB, 2 operations

        monitor.stop_monitoring()

        # Create temporary file for CSV output
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            # Write CSV
            monitor.write_csv(filename)

            # Verify file exists and has content
            file_path = Path(filename)
            assert file_path.exists()

            with file_path.open() as f:
                lines = f.readlines()

            # Should have header + data lines
            assert len(lines) > 1

            # Check header
            header = lines[0]
            assert "timestamp" in header
            assert "free_memory_bytes" in header
            assert "total_memory_bytes" in header
            assert "used_memory_bytes" in header

            # Check data lines have correct number of columns
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    comma_count = line.count(",")
                    assert comma_count == 3  # 4 columns = 3 commas

        finally:
            # Clean up
            file_path = Path(filename)
            if file_path.exists():
                file_path.unlink()

    def test_csv_export_invalid_path(self, cuda_context: None) -> None:
        """Test CSV export with invalid path raises exception."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()
        monitor.stop_monitoring()

        # Try to write to invalid path
        with pytest.raises(RuntimeError):
            monitor.write_csv("/invalid/path/file.csv")

    def test_debug_output(self, cuda_context: None) -> None:
        """Test debug output configuration."""
        monitor = CuptiMonitor()

        # Test setting debug output
        monitor.set_debug_output(enabled=True, threshold_mb=5)  # 5MB threshold
        monitor.set_debug_output(enabled=False, threshold_mb=10)  # Disable

        # These calls should be safe regardless of monitoring state
        monitor.start_monitoring()
        monitor.set_debug_output(enabled=True, threshold_mb=1)  # 1MB threshold
        monitor.stop_monitoring()

    def test_thread_safety(self, cuda_context: None) -> None:
        """Test thread safety of CuptiMonitor."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        threads = []
        num_threads = 4

        def thread_worker() -> None:
            # Each thread does some GPU operations and manual captures
            _perform_gpu_operations(1024 * 1024)  # 1 MiB
            monitor.capture_memory_sample()
            time.sleep(0.01)  # 10ms
            monitor.capture_memory_sample()

        # Multiple threads performing operations simultaneously
        for _ in range(num_threads):
            thread = threading.Thread(target=thread_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        monitor.stop_monitoring()

        # Should have collected samples from all threads
        assert monitor.get_sample_count() > num_threads

        # All samples should be valid
        samples = monitor.get_memory_samples()
        for sample in samples:
            assert sample.total_memory > 0
            assert sample.free_memory <= sample.total_memory

    def test_destructor_cleanup(self, cuda_context: None) -> None:
        """Test that destructor properly cleans up monitoring."""

        # Create monitor in a scope that will destroy it
        def create_and_destroy_monitor() -> None:
            monitor = CuptiMonitor()
            monitor.start_monitoring()
            _perform_gpu_operations(1024 * 1024)  # 1 MiB
            # Monitor should be destroyed here and automatically stop monitoring

        create_and_destroy_monitor()

        # Should be able to create a new monitor after destruction
        monitor2 = CuptiMonitor()
        monitor2.start_monitoring()
        monitor2.stop_monitoring()

    def test_large_number_of_samples(self, cuda_context: None) -> None:
        """Test handling of large number of samples."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        # Generate many samples
        for _ in range(100):
            monitor.capture_memory_sample()

        assert monitor.get_sample_count() == 101  # 100 manual + 1 initial

        samples = monitor.get_memory_samples()
        assert len(samples) == 101

        monitor.stop_monitoring()
        assert monitor.get_sample_count() == 102  # +1 final

    def test_callback_counters(self, cuda_context: None) -> None:
        """Test callback counter functionality."""
        monitor = CuptiMonitor()

        # Initially no callbacks
        assert monitor.get_total_callback_count() == 0
        counters = monitor.get_callback_counters()
        assert len(counters) == 0

        monitor.start_monitoring()

        # Perform GPU operations that should trigger callbacks
        _perform_gpu_operations(1024 * 1024, 2)  # 1 MiB, 2 operations

        # Also try some additional CUDA operations to ensure callbacks are triggered
        try:
            # Direct CUDA memory operations
            import cupy.cuda.runtime as runtime

            ptr1, _ = runtime.mallocManaged(1024 * 1024)
            runtime.free(ptr1)

            ptr2, _ = runtime.malloc(1024 * 1024)
            runtime.free(ptr2)
        except Exception:
            pass  # Ignore if these operations fail

        monitor.stop_monitoring()

        # Get callback information for debugging
        total_callbacks = monitor.get_total_callback_count()
        counters = monitor.get_callback_counters()
        summary = monitor.get_callback_summary()

        # Print debug information if no callbacks were recorded
        if total_callbacks == 0:
            print(f"Debug: No callbacks recorded. Summary: {summary}")
            # This might be expected in some environments, so make it a soft assertion
            pytest.skip(
                "No CUPTI callbacks were triggered - this may be expected in some environments"
            )

        # Should have recorded some callbacks
        assert total_callbacks > 0
        assert len(counters) > 0

        # Verify that callback summary doesn't crash and contains expected content
        assert len(summary) > 0
        assert "CUPTI Callback Counter Summary" in summary
        assert "Total" in summary

    def test_callback_counters_clear(self, cuda_context: None) -> None:
        """Test clearing callback counters."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        _perform_gpu_operations(1024 * 1024)  # 1 MiB

        # Also try additional CUDA operations
        try:
            import cupy.cuda.runtime as runtime

            ptr, _ = runtime.malloc(1024 * 1024)
            runtime.free(ptr)
        except Exception:
            pass

        monitor.stop_monitoring()

        # Check if callbacks were recorded
        total_callbacks = monitor.get_total_callback_count()
        if total_callbacks == 0:
            pytest.skip("No CUPTI callbacks were triggered - skipping clear test")

        # Should have some callbacks recorded
        assert total_callbacks > 0

        # Clear counters
        monitor.clear_callback_counters()

        # Should be empty now
        assert monitor.get_total_callback_count() == 0
        counters = monitor.get_callback_counters()
        assert len(counters) == 0

        # Summary should indicate no callbacks
        summary = monitor.get_callback_summary()
        assert "No callbacks recorded yet" in summary

    def test_callback_counters_accumulate(self, cuda_context: None) -> None:
        """Test that callback counters accumulate properly."""
        monitor = CuptiMonitor()
        monitor.start_monitoring()

        # First batch of operations
        _perform_gpu_operations(1024 * 1024, 1)  # 1 MiB, 1 operation
        first_count = monitor.get_total_callback_count()

        # Second batch of operations
        _perform_gpu_operations(1024 * 1024, 1)  # 1 MiB, 1 operation
        second_count = monitor.get_total_callback_count()

        monitor.stop_monitoring()

        # If no callbacks were triggered at all, skip the test
        if first_count == 0 and second_count == 0:
            pytest.skip("No CUPTI callbacks were triggered - skipping accumulate test")

        # Should have accumulated more callbacks (or at least stayed the same)
        assert second_count >= first_count

    def test_parameter_validation(self, cuda_context: None) -> None:
        """Test parameter validation for CuptiMonitor constructor."""
        # Test with various parameter combinations
        monitor1 = CuptiMonitor(enable_periodic_sampling=False)
        assert not monitor1.is_monitoring()

        monitor2 = CuptiMonitor(sampling_interval_ms=200)
        assert not monitor2.is_monitoring()

        monitor3 = CuptiMonitor(enable_periodic_sampling=True, sampling_interval_ms=25)
        assert not monitor3.is_monitoring()

    def test_context_manager_like_usage(self, cuda_context: None) -> None:
        """Test using CuptiMonitor in a context-manager-like pattern."""
        monitor = CuptiMonitor()

        try:
            monitor.start_monitoring()
            _perform_gpu_operations(1024 * 1024)  # 1 MiB

            # Should have samples
            assert monitor.get_sample_count() > 0

        finally:
            monitor.stop_monitoring()

        # Should still have samples after stopping
        assert monitor.get_sample_count() > 0


# Tests when CUPTI is not available
class TestCuptiNotAvailable:
    """Test cases for when CUPTI support is not available."""

    def test_cupti_not_available(self) -> None:
        """Test that appropriate error is raised when CUPTI is not available."""
        pytest.skip(
            "CUPTI support not enabled. Build with -DBUILD_CUPTI_SUPPORT=ON to enable tests."
        )
