/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <limits>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
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

}  // namespace

PinnedMemoryResource::PinnedMemoryResource(
    int numa_id, PinnedPoolProperties pool_properties
)
    : shared_base([&] {
          RAPIDSMPF_EXPECTS(
              is_pinned_memory_resources_supported(),
              "Pinned host memory is not supported on this system. "
              "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
              " is one of the requirements, but additional platform or driver "
              "constraints may apply. If needed, use `PinnedMemoryResource::Disabled` "
              "to disable pinned host memory, noting that this may significantly "
              "degrade spilling performance.",
              std::invalid_argument
          );
          return cuda::mr::make_shared_resource<
              detail::RmmResourceAdaptorImpl<cuda::pinned_memory_pool>>(
              std::in_place, numa_id, get_memory_pool_properties(pool_properties)
          );
      }()),
      pool_properties_{std::move(pool_properties)} {}

std::optional<PinnedMemoryResource> PinnedMemoryResource::make_if_available(
    int numa_id, PinnedPoolProperties pool_properties
) {
    if (is_pinned_memory_resources_supported()) {
        return PinnedMemoryResource{numa_id, std::move(pool_properties)};
    }
    return PinnedMemoryResource::Disabled;
}

std::optional<PinnedMemoryResource> PinnedMemoryResource::from_options(
    config::Options options
) {
    bool const pinned_memory = options.get<bool>("pinned_memory", [](auto const& s) {
        return s.empty() ? EnabledByDefault : parse_string<bool>(s);
    });

    if (pinned_memory && is_pinned_memory_resources_supported()) {
        auto const host_memory_per_gpu = get_host_memory_per_gpu();
        PinnedPoolProperties pool_properties{
            .initial_pool_size = options.get<size_t>(
                "pinned_initial_pool_size",
                [&](auto const& s) {
                    return parse_nbytes_or_percent(
                        s.empty() ? DefaultInitiPoolSizeFactor : s,
                        safe_cast<double>(host_memory_per_gpu)
                    );
                }
            ),
            .max_pool_size = options.get<std::optional<size_t>>(
                "pinned_max_pool_size", [&](auto const& s) {
                    return parse_nbytes_or_percent(
                        s.empty() ? DefaultMaxPoolSizeFactor : s,
                        safe_cast<double>(host_memory_per_gpu)
                    );
                }
            )
        };
        return PinnedMemoryResource{get_current_numa_node(), std::move(pool_properties)};
    }
    return PinnedMemoryResource::Disabled;
}

std::function<std::int64_t()> PinnedMemoryResource::get_memory_available_cb() const {
    auto const max_pool_size = pool_properties_.max_pool_size.value_or(0);
    if (max_pool_size > 0) {
        auto const limit = safe_cast<std::int64_t>(max_pool_size);
        return [tracker = *this, limit]() {
            return limit - tracker.get().current_allocated();
        };
    }
    return std::numeric_limits<std::int64_t>::max;
}

}  // namespace rapidsmpf
