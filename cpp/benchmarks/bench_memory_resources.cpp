/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <vector>

#include <benchmark/benchmark.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>

namespace {
class HostMemoryResource {
  public:
    virtual ~HostMemoryResource() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr, size_t bytes) noexcept = 0;
};

class NewDeleteHostMemoryResource final : public HostMemoryResource {
  public:
    void* allocate(size_t bytes) override {
        return ::operator new(bytes);
    }

    void deallocate(void* ptr, size_t) noexcept override {
        ::operator delete(ptr);
    }
};

class PinnedHostMemoryResource final : public HostMemoryResource {
  public:
    void* allocate(size_t bytes) override {
        return mr.allocate_sync(bytes);
    }

    void deallocate(void* ptr, size_t bytes) noexcept override {
        mr.deallocate_sync(ptr, bytes);
    }

    rmm::mr::pinned_host_memory_resource mr{};
};

class CcclPinnedHostMemoryResource final : public HostMemoryResource {
  public:
    CcclPinnedHostMemoryResource() : p_mr{p_pool} {}

    void* allocate(size_t bytes) override {
        return p_mr.allocate_sync(bytes);
    }

    void deallocate(void* ptr, size_t bytes) noexcept override {
        p_mr.deallocate_sync(ptr, bytes);
    }

    rapidsmpf::PinnedMemoryPool p_pool{};
    rapidsmpf::PinnedMemoryResource p_mr;
};

enum ResourceType : int {
    NEW_DELETE = 0,
    PINNED = 1,
    CCCL_PINNED = 2,
};

static constexpr std::array<std::string, 3> ResourceTypeStr{
    "new_delete", "pinned", "cccl_pinned"
};
}  // namespace

// Helper function to create a memory resource based on type
std::unique_ptr<HostMemoryResource> create_host_memory_resource(
    const ResourceType& resource_type
) {
    switch (resource_type) {
    case ResourceType::NEW_DELETE:
        return std::make_unique<NewDeleteHostMemoryResource>();
    case ResourceType::PINNED:
        return std::make_unique<PinnedHostMemoryResource>();
    case ResourceType::CCCL_PINNED:
        return std::make_unique<CcclPinnedHostMemoryResource>();
    default:
        RAPIDSMPF_FAIL("Unknown memory resource type");
    }
}

// Benchmark for allocation
static void BM_Allocate(benchmark::State& state) {
    const auto allocation_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto mr = create_host_memory_resource(resource_type);

    for (auto _ : state) {
        void* ptr = mr->allocate(allocation_size);
        benchmark::DoNotOptimize(ptr);

        state.PauseTiming();
        mr->deallocate(ptr, allocation_size);
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(allocation_size));
    state.SetLabel("allocate: " + ResourceTypeStr[resource_type]);
}

// Benchmark for deallocation
static void BM_Deallocate(benchmark::State& state) {
    const auto allocation_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto mr = create_host_memory_resource(resource_type);

    for (auto _ : state) {
        state.PauseTiming();
        void* ptr = mr->allocate(allocation_size);
        state.ResumeTiming();

        mr->deallocate(ptr, allocation_size);
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(allocation_size));
    state.SetLabel("deallocate: " + ResourceTypeStr[resource_type]);
}

static constexpr int64_t kNumCopies = 8;

// Benchmark for device to host transfer. This benchmark allocates a device buffer and a
// host buffer, then copies the device buffer to the host buffer kNumCopies times.
static void BM_DeviceToHostCopy(benchmark::State& state) {
    const auto transfer_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Allocate device memory
    auto device_buffer = rmm::device_buffer(transfer_size, stream, device_mr.get());
    // Initialize device memory
    RAPIDSMPF_CUDA_TRY(cudaMemset(device_buffer.data(), 0, transfer_size));

    // Allocate host memory and copy from device
    void* host_ptr = host_mr->allocate(transfer_size);

    benchmark::DoNotOptimize(device_buffer);
    benchmark::DoNotOptimize(host_ptr);

    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    for (auto _ : state) {
        for (size_t i = 0; i < kNumCopies; ++i) {
            cudaMemcpyAsync(
                static_cast<char*>(host_ptr),
                device_buffer.data(),
                transfer_size,
                cudaMemcpyDeviceToHost,
                stream
            );
        }
        RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    }
    // Cleanup
    host_mr->deallocate(host_ptr, transfer_size);

    state.SetBytesProcessed(
        int64_t(state.iterations()) * int64_t(transfer_size) * kNumCopies
    );
    state.SetLabel("memcpy device to host: " + ResourceTypeStr[resource_type]);
}

// Benchmark for host to device transfer. This benchmark allocates a host buffer and a
// device buffer, then copies the host buffer to the device buffer kNumCopies times.
static void BM_HostToDeviceCopy(benchmark::State& state) {
    const auto transfer_size = static_cast<size_t>(state.range(0));
    const auto resource_type = static_cast<ResourceType>(state.range(1));

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Allocate host memory and initialize
    void* host_ptr = host_mr->allocate(transfer_size);
    memset(host_ptr, 0, transfer_size);

    // Allocate device memory and copy from host
    auto device_buffer = rmm::device_buffer(transfer_size, stream, device_mr.get());

    benchmark::DoNotOptimize(device_buffer);
    benchmark::DoNotOptimize(host_ptr);

    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        for (size_t i = 0; i < kNumCopies; ++i) {
            cudaMemcpyAsync(
                static_cast<char*>(device_buffer.data()),
                host_ptr,
                transfer_size,
                cudaMemcpyHostToDevice,
                stream
            );
        }
        RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    }
    // Cleanup
    host_mr->deallocate(host_ptr, transfer_size);

    state.SetBytesProcessed(
        int64_t(state.iterations()) * int64_t(transfer_size) * kNumCopies
    );
    state.SetLabel("memcpy host to device: " + ResourceTypeStr[resource_type]);
}

// Custom argument generator for the benchmark
void CustomArguments(benchmark::internal::Benchmark* b) {
    // Test different allocation sizes
    for (auto size : {1 << 10, 500 << 10, 1 << 20, 500 << 20, 1 << 30}) {
        // Test both memory resource types
        for (auto resource_type :
             {ResourceType::NEW_DELETE, ResourceType::PINNED, ResourceType::CCCL_PINNED})
        {
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
