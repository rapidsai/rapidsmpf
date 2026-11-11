/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <limits>
#include <memory>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

#include <cuda/stream>

#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/system_info.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace {

/**
 * @brief Enable Transparent Huge Pages (THP) for a memory region.
 *
 * Attempts to mark the specified memory region as eligible for Transparent Huge Pages
 * (THP) using `madvise(MADV_HUGEPAGE)`. This is a best-effort optimization that can
 * improve device to host memory transfer performance for sufficiently large buffers.
 * See <https://github.com/rapidsai/cudf/pull/13914>.
 *
 * @param ptr Pointer to the start of the memory region.
 * @param size Size of the region in bytes.
 */
void enable_hugepage_for_region(void* ptr, std::size_t size) {
    if (size < (1u << 22u)) {  // smaller than 4 MiB, skip
        return;
    }
#ifdef MADV_HUGEPAGE
    static auto const pagesize = safe_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    if (std::align(pagesize, pagesize, ptr, size)) {
        // Best effort, we ignore errors. On older kernels this may fail or be a no-op.
        ::madvise(ptr, size, MADV_HUGEPAGE);
    }
#endif
}

}  // namespace

void* detail::HostMemoryResourceImpl::allocate(
    cuda::stream_ref, std::size_t size, std::size_t alignment
) {
    void* ret = ::operator new(size, std::align_val_t{alignment});
    enable_hugepage_for_region(ret, size);
    return ret;
}

void detail::HostMemoryResourceImpl::deallocate(
    cuda::stream_ref stream, void* ptr, std::size_t, std::size_t alignment
) noexcept {
    stream.sync();
    ::operator delete(ptr, std::align_val_t{alignment});
}

std::optional<std::int64_t> host_limit_from_options(config::Options options) {
    return options.get<std::optional<std::int64_t>>(
        "spill_host_limit", [](auto const& s) -> std::optional<std::int64_t> {
            auto const value = parse_optional(s);
            if (!value.has_value()) {
                return std::nullopt;
            }
            auto const total = safe_cast<double>(get_host_memory_per_gpu());
            auto const limit = safe_cast<std::int64_t>(rmm::align_down(
                parse_nbytes_or_percent(*value, total), rmm::CUDA_ALLOCATION_ALIGNMENT
            ));
            RAPIDSMPF_EXPECTS(
                limit > 0,
                "spill_host_limit must be greater than zero",
                std::invalid_argument
            );
            return limit;
        }
    );
}
}  // namespace rapidsmpf
