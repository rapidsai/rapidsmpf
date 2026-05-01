/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

#include <benchmark/benchmark.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>

using rapidsmpf::safe_cast;

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

class NewDelete {
  public:
    void* allocate_sync(
        std::size_t size, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return ::operator new(size, std::align_val_t{alignment});
    }

    void deallocate_sync(
        void* ptr, std::size_t, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        ::operator delete(ptr, std::align_val_t{alignment});
    }

    void* allocate(
        rmm::cuda_stream_view,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return allocate_sync(size, alignment);
    }

    void deallocate(
        rmm::cuda_stream_view,
        void* ptr,
        std::size_t,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        deallocate_sync(ptr, alignment);
    }

    bool operator==(NewDelete const&) const noexcept {
        return true;
    }

    bool operator!=(NewDelete const&) const noexcept {
        return false;
    }

    friend void get_property(NewDelete const&, cuda::mr::host_accessible) noexcept {}
};

// Helper function to create a type-erased host memory resource.
cuda::mr::any_resource<cuda::mr::host_accessible> create_host_memory_resource(
    ResourceType const& resource_type
) {
    switch (resource_type) {
    case ResourceType::NEW_DELETE:
        return NewDelete{};
    case ResourceType::HOST_MEMORY_RESOURCE:
        return rapidsmpf::HostMemoryResource{};
    case ResourceType::PINNED_MEMORY_RESOURCE:
        {
            auto mr = rapidsmpf::PinnedMemoryResource::make_if_available();
            RAPIDSMPF_EXPECTS(
                mr.has_value(),
                "pinned memory is not supported on this system",
                std::runtime_error
            );
            return *mr;
        }
    default:
        RAPIDSMPF_FAIL("Unknown memory resource type");
    }
}

void BM_Allocate(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const allocation_size = static_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto mr = create_host_memory_resource(resource_type);
    for (auto _ : state) {
        void* ptr = mr.allocate(stream, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        benchmark::DoNotOptimize(ptr);
        stream.synchronize();

        state.PauseTiming();
        mr.deallocate(stream, ptr, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        stream.synchronize();
        state.ResumeTiming();
    }

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(allocation_size)
    );
    state.SetLabel(
        "allocate: " + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

void BM_Deallocate(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const allocation_size = safe_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto mr = create_host_memory_resource(resource_type);
    for (auto _ : state) {
        state.PauseTiming();
        void* ptr = mr.allocate(stream, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        stream.synchronize();
        state.ResumeTiming();

        mr.deallocate(stream, ptr, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        stream.synchronize();
    }

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(allocation_size)
    );
    state.SetLabel(
        "deallocate: " + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

void BM_DeviceToHostCopyInclAlloc(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const transfer_size = static_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Allocate device memory
    auto src = rmm::device_buffer(transfer_size, stream, *device_mr);
    // Initialize src to avoid optimization removal
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(src.data(), 0xAB, transfer_size, stream));
    stream.synchronize();

    for (auto _ : state) {
        void* dst =
            host_mr.allocate(stream, transfer_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        RAPIDSMPF_CUDA_TRY(
            rapidsmpf::cuda_memcpy_async(dst, src.data(), transfer_size, stream)
        );
        stream.synchronize();

        state.PauseTiming();
        host_mr.deallocate(stream, dst, transfer_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        state.ResumeTiming();
    }

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(transfer_size)
    );
    state.SetLabel(
        "memcpy device to host (incl. alloc): "
        + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

template <typename T>
void bench_copy(
    benchmark::State& state,
    T& mr,
    void const* src,
    std::size_t size,
    rmm::cuda_stream_view stream
) {
    for (auto _ : state) {
        state.PauseTiming();
        void* dst = mr.allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        stream.synchronize();
        state.ResumeTiming();

        RAPIDSMPF_CUDA_TRY(rapidsmpf::cuda_memcpy_async(dst, src, size, stream));
        stream.synchronize();

        state.PauseTiming();
        mr.deallocate(stream, dst, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        state.ResumeTiming();
    }
}

void BM_DeviceToHostCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const transfer_size = static_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    auto src = rmm::device_buffer(transfer_size, stream, *device_mr);
    // Initialize src to avoid optimization removal
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(src.data(), 0xAB, transfer_size, stream));

    bench_copy(state, host_mr, src.data(), transfer_size, stream);

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(transfer_size)
    );
    state.SetLabel(
        "memcpy device to host: "
        + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

void BM_HostToDeviceCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const transfer_size = static_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto host_mr = create_host_memory_resource(resource_type);
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Allocate host memory and initialize
    void* host_ptr =
        host_mr.allocate(stream, transfer_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    memset(host_ptr, 0, transfer_size);

    // Allocate device memory and copy from host
    auto src = rmm::device_buffer(transfer_size, stream, *device_mr);

    bench_copy(state, host_mr, src.data(), transfer_size, stream);

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(transfer_size)
    );
    state.SetLabel(
        "memcpy host to device: "
        + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

void BM_HostToHostCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const transfer_size = static_cast<std::size_t>(state.range(0));
    auto const resource_type = static_cast<ResourceType>(state.range(1));

    if (resource_type == ResourceType::PINNED_MEMORY_RESOURCE
        && !rapidsmpf::is_pinned_memory_resources_supported())
    {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    auto host_mr = create_host_memory_resource(resource_type);

    void* src = host_mr.allocate(stream, transfer_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

    // Initialize src to avoid optimization elimination
    std::memset(src, 0xAB, transfer_size);

    bench_copy(state, host_mr, src, transfer_size, stream);

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(transfer_size)
    );
    state.SetLabel(
        "memcpy host to host: " + ResourceTypeStr[static_cast<std::size_t>(resource_type)]
    );
}

void BM_DeviceToDeviceCopy(benchmark::State& state) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const transfer_size = static_cast<std::size_t>(state.range(0));

    // Device MR, independent of host resource type
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    rmm::device_buffer src(transfer_size, stream, *device_mr);

    // Initialize src to avoid optimization removal
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(src.data(), 0xAB, transfer_size, stream));

    bench_copy(state, *device_mr, src.data(), transfer_size, stream);

    state.SetBytesProcessed(
        safe_cast<std::int64_t>(state.iterations())
        * safe_cast<std::int64_t>(transfer_size)
    );
    state.SetLabel("memcpy device to device: rmm::mr::cuda_memory_resource");
}

// Custom argument generator for the benchmark
void CustomArguments(benchmark::Benchmark* b) {
    // Test different allocation sizes
    for (auto size : {1 << 10, 500 << 10, 1 << 20, 500 << 20, 1 << 30}) {
        // Test all memory resource types
        for (auto resource_type : RESOURCE_TYPES) {
            b->Args({size, resource_type});
        }
    }
}

// Register the benchmarks

BENCHMARK(BM_DeviceToHostCopyInclAlloc)
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

BENCHMARK(BM_Allocate)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Deallocate)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_HostToHostCopy)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_DeviceToDeviceCopy)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

// First large allocation: impact of initial_pool_size (with vs without initial size).
void BM_PinnedFirstAlloc_InitialPoolSize(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    // Ensure CUDA device context is initialized (required for pinned memory pools).
    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    auto const allocation_size = static_cast<std::size_t>(state.range(0)) << 20;
    auto const primed = static_cast<bool>(state.range(1));

    // set initial pool size to allocation size if primed, 0 otherwise
    rapidsmpf::PinnedPoolProperties props{
        .initial_pool_size = primed ? allocation_size : 0
    };

    for (auto _ : state) {
        state.PauseTiming();
        auto mr = rapidsmpf::PinnedMemoryResource::make_if_available(
            rapidsmpf::get_current_numa_node(), props
        );
        state.ResumeTiming();
        void* ptr = mr->allocate(stream, allocation_size);
        stream.synchronize();
        state.PauseTiming();
        mr->deallocate(stream, ptr, allocation_size);
        stream.synchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(allocation_size));
    state.counters["initial_pool_size"] =
        static_cast<double>(primed ? allocation_size : 0);
}

void PinnedFirstAlloc_InitialPoolSize_Args(benchmark::Benchmark* b) {
    for (auto size : {1, 256, 1024}) {  // in MB
        b->Args({size, 1});  // primed
        b->Args({size, 0});  // no priming
    }
}

BENCHMARK(BM_PinnedFirstAlloc_InitialPoolSize)
    ->Apply(PinnedFirstAlloc_InitialPoolSize_Args)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
