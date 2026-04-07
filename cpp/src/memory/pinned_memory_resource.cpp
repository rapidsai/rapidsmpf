/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <limits>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/mr/pinned_host_memory_resource.hpp>
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
      pool_{make_pinned_memory_pool(numa_id, pool_properties_)},
      pool_tracker_{cuda::mr::make_shared_resource<RmmResourceAdaptor>(pool_)} {}

PinnedMemoryResource::PinnedMemoryResource(
    int numa_id,
    PinnedPoolProperties pool_properties,
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t capacity,
    std::size_t initial_npools
)
    : pool_properties_{std::move(pool_properties)},
      pool_{make_pinned_memory_pool(numa_id, pool_properties_)},
      pool_tracker_{cuda::mr::make_shared_resource<RmmResourceAdaptor>(pool_)},
      fixed_size_host_mr_{std::make_shared<FixedSizedHostMemoryResource>(
          numa_id,
          *pool_tracker_,
          capacity,
          capacity,
          block_size,
          pool_size,
          initial_npools
      )} {}

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
    bool const pinned_memory_fixed_size =
        options.get<bool>("pinned_memory_fixed_size", [](auto const& s) {
            return parse_string<bool>(s.empty() ? "False" : s);
        });
    if (is_pinned_memory_resources_supported()
        && (pinned_memory || pinned_memory_fixed_size))
    {
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

        if (pinned_memory_fixed_size) {
            auto const fixed_size_block_size = options.get<size_t>(
                "pinned_memory_fixed_size_block_size", [](auto const& s) {
                    return parse_nbytes_unsigned(s.empty() ? "1MiB" : s);
                }
            );

            return PinnedMemoryResource::make_fixed_sized_if_available(
                get_current_numa_node(), std::move(pool_properties), fixed_size_block_size
            );
        } else {
            return PinnedMemoryResource::make_if_available(
                get_current_numa_node(), std::move(pool_properties)
            );
        }
    }

    return PinnedMemoryResource::Disabled;
}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate(
    rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment
) {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ == nullptr, "allocate called with fixed size mr available"
    );
    return pool_tracker_->allocate(stream, bytes, alignment);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment
) noexcept {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ == nullptr, "deallocate called with fixed size mr available"
    );
    pool_tracker_tracker_->deallocate(stream, ptr, bytes, alignment);
}

void* PinnedMemoryResource::allocate_sync(std::size_t bytes, std::size_t alignment) {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ == nullptr,
        "allocate_sync called with fixed size mr available"
    );
    return pool_tracker_->allocate_sync(bytes, alignment);
}

void PinnedMemoryResource::deallocate_sync(
    void* ptr, std::size_t bytes, std::size_t alignment
) {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ == nullptr,
        "deallocate_sync called with fixed size mr available"
    );
    pool_tracker_->deallocate_sync(ptr, bytes, alignment);
}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::make_fixed_sized_if_available(
    int numa_id,
    PinnedPoolProperties pool_properties,
    std::size_t block_size,
    std::size_t pool_size
) {
    if (!is_pinned_memory_resources_supported()) {
        return PinnedMemoryResource::Disabled;
    }
    size_t const capacity =
        pool_properties.max_pool_size.value_or(get_numa_node_host_memory(numa_id));

    size_t const initial_npools = std::max(
        cucascade::memory::fixed_size_host_memory_resource::default_initial_number_pools,
        pool_properties.initial_pool_size / (block_size * pool_size)
    );

    return std::shared_ptr<PinnedMemoryResource>(new PinnedMemoryResource(
        numa_id,
        std::move(pool_properties),
        block_size,
        pool_size,
        capacity,
        initial_npools
    ));
}

PinnedMemoryResource::FixedSizedBlocksAllocation
PinnedMemoryResource::allocate_fixed_sized(std::size_t size) {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ != nullptr,
        "fixed-size host memory resource not initialized; "
        "use make_fixed_sized_if_available to create this resource",
        std::invalid_argument
    );
    return fixed_size_host_mr_->allocate_multiple_blocks(size);
}

std::function<std::int64_t()> PinnedMemoryResource::get_memory_available_cb() const {
    auto const max_pool_size = pool_properties_.max_pool_size.value_or(0);
    if (max_pool_size > 0) {
        return LimitAvailableMemory{
            &pool_tracker_.get(), safe_cast<std::int64_t>(max_pool_size)
        };
    }
    return std::numeric_limits<std::int64_t>::max;
}

bool PinnedMemoryResource::is_equal(HostMemoryResource const& other) const noexcept {
    auto const* o = dynamic_cast<PinnedMemoryResource const*>(&other);
    return o != nullptr && pool_ == o->pool_
           && fixed_size_host_mr_ == o->fixed_size_host_mr_;
}

}  // namespace rapidsmpf
