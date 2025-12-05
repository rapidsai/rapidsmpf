/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

#include <benchmark/benchmark.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>

enum ResourceType : int {
    NEW_DELETE = 0,
    HOST_MEMORY_RESOURCE = 1,
    PINNED_MEMORY_RESOURCE = 2,
};

constexpr std::array<ResourceType, 3> RESOURCE_TYPES{
    ResourceType::NEW_DELETE,
    ResourceType::HOST_MEMORY_RESOURCE,
    ResourceType::PINNED_MEMORY_RESOURCE
};

std::array<std::string, 3> const ResourceTypeStr{
    "NewDelete", "HostMemoryResource", "PinnedMemoryResource"
};

class NewDelete final : public rapidsmpf::HostMemoryResource {
  public:
    void* allocate(
        rmm::cuda_stream_view,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) override {
        return ::operator new(size, std::align_val_t{alignment});
    }

    void deallocate(
        rmm::cuda_stream_view,
        void* ptr,
        std::size_t,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept override {
        ::operator delete(ptr, std::align_val_t{alignment});
    }
};

// Helper function to create a memory resource based on type
std::unique_ptr<rapidsmpf::HostMemoryResource> create_host_memory_resource(
    const ResourceType& resource_type
) {
    switch (resource_type) {
    case ResourceType::NEW_DELETE:
        return std::make_unique<NewDelete>();
    case ResourceType::HOST_MEMORY_RESOURCE:
        return std::make_unique<rapidsmpf::HostMemoryResource>();
    case ResourceType::PINNED_MEMORY_RESOURCE:
        return std::make_unique<rapidsmpf::PinnedMemoryResource>();
    default:
        RAPIDSMPF_FAIL("Unknown memory resource type");
    }
}

// Benchmark for allocation
static void BM_Allocate(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    const auto allocation_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto mr = create_host_memory_resource(resource_type);

    for (auto _ : state) {
        void* ptr = mr->allocate(stream, allocation_size);
        benchmark::DoNotOptimize(ptr);
        stream.synchronize();

        state.PauseTiming();
        mr->deallocate(stream, ptr, allocation_size);
        stream.synchronize();
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(allocation_size));
    state.SetLabel(
        "allocate: " + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

// Benchmark for deallocation
static void BM_Deallocate(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    const auto allocation_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto mr = create_host_memory_resource(resource_type);

    for (auto _ : state) {
        state.PauseTiming();
        void* ptr = mr->allocate(stream, allocation_size);
        stream.synchronize();
        state.ResumeTiming();

        mr->deallocate(stream, ptr, allocation_size);
        stream.synchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(allocation_size));
    state.SetLabel(
        "deallocate: " + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

static constexpr int64_t kNumCopies = 8;

// Benchmark for device to host transfer. This benchmark allocates a device buffer and a
// host buffer, then copies the device buffer to the host buffer kNumCopies times.
static void BM_DeviceToHostCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    const auto transfer_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Allocate device memory
    auto device_buffer = rmm::device_buffer(transfer_size, stream, device_mr.get());
    // Initialize device memory
    RAPIDSMPF_CUDA_TRY(cudaMemset(device_buffer.data(), 0, transfer_size));

    // Allocate host memory and copy from device
    void* host_ptr = host_mr->allocate(stream, transfer_size);

    benchmark::DoNotOptimize(device_buffer);
    benchmark::DoNotOptimize(host_ptr);

    stream.synchronize();
    for (auto _ : state) {
        for (size_t i = 0; i < kNumCopies; ++i) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                static_cast<char*>(host_ptr),
                device_buffer.data(),
                transfer_size,
                cudaMemcpyDeviceToHost,
                stream
            ));
        }
        stream.synchronize();
    }
    // Cleanup
    host_mr->deallocate(stream, host_ptr, transfer_size);

    state.SetBytesProcessed(
        int64_t(state.iterations()) * int64_t(transfer_size) * kNumCopies
    );
    state.SetLabel(
        "memcpy device to host: "
        + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

// Benchmark for host to device transfer. This benchmark allocates a host buffer and a
// device buffer, then copies the host buffer to the device buffer kNumCopies times.
static void BM_HostToDeviceCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    const auto transfer_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Allocate host memory and initialize
    void* host_ptr = host_mr->allocate(stream, transfer_size);
    memset(host_ptr, 0, transfer_size);

    // Allocate device memory and copy from host
    auto device_buffer = rmm::device_buffer(transfer_size, stream, device_mr.get());

    benchmark::DoNotOptimize(device_buffer);
    benchmark::DoNotOptimize(host_ptr);

    stream.synchronize();
    for (auto _ : state) {
        for (size_t i = 0; i < kNumCopies; ++i) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                static_cast<char*>(device_buffer.data()),
                host_ptr,
                transfer_size,
                cudaMemcpyHostToDevice,
                stream
            ));
        }
        stream.synchronize();
    }
    // Cleanup
    host_mr->deallocate(stream, host_ptr, transfer_size);

    state.SetBytesProcessed(
        int64_t(state.iterations()) * int64_t(transfer_size) * kNumCopies
    );
    state.SetLabel(
        "memcpy host to device: "
        + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

// Custom argument generator for the benchmark
void CustomArguments(benchmark::internal::Benchmark* b) {
    // Test different allocation sizes
    for (auto size : {1 << 10, 500 << 10, 1 << 20, 500 << 20, 1 << 30}) {
        // Test all memory resource types
        for (auto resource_type : RESOURCE_TYPES) {
            b->Args({size, resource_type});
        }
    }
}

// Register the benchmarks
BENCHMARK(BM_Allocate)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Deallocate)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_DeviceToHostCopy)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_HostToDeviceCopy)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
