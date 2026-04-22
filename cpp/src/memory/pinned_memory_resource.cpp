/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <limits>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

namespace {
cuda::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const& pool_properties
) {
    return cuda::memory_pool_properties{
        // It was observed that priming async device pools have little effect on
        // performance. See <https://github.com/rapidsai/rmm/issues/1931>. However,
        // initial allocations and warming up the pool have a significant impact on
        // pinned memory pool performance.
        .initial_pool_size = pool_properties.initial_pool_size,
        // Before <https://github.com/NVIDIA/cccl/pull/6718>, the default
        // `release_threshold` was 0, which defeats the purpose of having a pool. We
        // now set it so the pool never releases unused pinned memory.
        .release_threshold = std::numeric_limits<std::size_t>::max(),
        // This defines how the allocations can be exported (IPC). See the docs of
        // `cudaMemPoolCreate` in <https://docs.nvidia.com/cuda/cuda-runtime-api>.
        .allocation_handle_type = ::cudaMemAllocationHandleType::cudaMemHandleTypeNone,
        .max_pool_size = pool_properties.max_pool_size.value_or(0),
    };
}

cuda::mr::shared_resource<cuda::pinned_memory_pool> make_pinned_memory_pool(
    int numa_id, PinnedPoolProperties const& props
) {
    RAPIDSMPF_EXPECTS(
        is_pinned_memory_resources_supported(),
        "Pinned host memory is not supported on this system. "
        "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        " is one of the requirements, but additional platform or driver constraints may "
        "apply. If needed, use `PinnedMemoryResource::Disabled` to disable pinned host "
        "memory, noting that this may significantly degrade spilling performance.",
        std::invalid_argument
    );
    return cuda::mr::make_shared_resource<cuda::pinned_memory_pool>(
        numa_id, get_memory_pool_properties(props)
    );
}
}  // namespace

PinnedMemoryResource::PinnedMemoryResource(
    int numa_id, PinnedPoolProperties pool_properties
)
    : pool_properties_{std::move(pool_properties)},
      pool_tracker_{make_pinned_memory_pool(numa_id, pool_properties_)} {}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::make_if_available(
    int numa_id, PinnedPoolProperties pool_properties
) {
    if (is_pinned_memory_resources_supported()) {
        return std::make_shared<rapidsmpf::PinnedMemoryResource>(
            numa_id, std::move(pool_properties)
        );
    }
    return PinnedMemoryResource::Disabled;
}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::from_options(
    config::Options options
) {
    bool const pinned_memory = options.get<bool>("pinned_memory", [](auto const& s) {
        return parse_string<bool>(s.empty() ? "True" : s);
    });
    if (pinned_memory && is_pinned_memory_resources_supported()) {
        PinnedPoolProperties pool_properties{
            .initial_pool_size = options.get<size_t>(
                "pinned_initial_pool_size",
                [](auto const& s) { return s.empty() ? 0 : parse_nbytes_unsigned(s); }
            ),
            .max_pool_size = options.get<std::optional<size_t>>(
                "pinned_max_pool_size", [](auto const& s) -> std::optional<size_t> {
                    auto parsed = parse_optional(s);
                    if (parsed.has_value() && !parsed->empty()) {
                        return parse_nbytes_unsigned(*parsed);
                    }
                    return std::nullopt;
                }
            )
        };
        return PinnedMemoryResource::make_if_available(
            get_current_numa_node(), std::move(pool_properties)
        );
    }
    return PinnedMemoryResource::Disabled;
}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate(
    rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment
) {
    return pool_tracker_->allocate(stream, bytes, alignment);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment
) noexcept {
    pool_tracker_->deallocate(stream, ptr, bytes, alignment);
}

std::function<std::int64_t()> PinnedMemoryResource::get_memory_available_cb() const {
    auto const max_pool_size = pool_properties_.max_pool_size.value_or(0);
    if (max_pool_size > 0) {
        return LimitAvailableMemory{
            pool_tracker_, safe_cast<std::int64_t>(max_pool_size)
        };
    }
    return std::numeric_limits<std::int64_t>::max;
}

bool PinnedMemoryResource::is_equal(HostMemoryResource const& other) const noexcept {
    auto const* o = dynamic_cast<PinnedMemoryResource const*>(&other);
    return o != nullptr && pool_tracker_ == o->pool_tracker_;
}

}  // namespace rapidsmpf
