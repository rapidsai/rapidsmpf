/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <memory>

#include <sys/mman.h>
#include <unistd.h>

#include <rapidsmpf/memory/host_memory_resource.hpp>

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
    static auto const pagesize = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    if (std::align(pagesize, pagesize, ptr, size)) {
        // Best effort, we ignore errors. On older kernels this may fail or be a no-op.
        ::madvise(ptr, size, MADV_HUGEPAGE);
    }
#endif
}

}  // namespace

void* HostMemoryResource::allocate(
    rmm::cuda_stream_view, std::size_t size, std::size_t alignment
) {
    void* ret = ::operator new(size, std::align_val_t{alignment});
    enable_hugepage_for_region(ret, size);
    return ret;
}

void HostMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, std::size_t, std::size_t alignment
) noexcept {
    stream.synchronize();
    ::operator delete(ptr, std::align_val_t{alignment});
}
}  // namespace rapidsmpf
