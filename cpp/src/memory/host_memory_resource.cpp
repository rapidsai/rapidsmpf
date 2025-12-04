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
 * improve device, host memory transfer performance for sufficiently large buffers.
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
    auto const pagesize = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    void* addr = ptr;
    auto length = size;

    if (std::align(pagesize, pagesize, addr, length)) {
        // Best effort, we ignore errors. On older kernels this may fail or be a no-op.
        ::madvise(addr, length, MADV_HUGEPAGE);
    }
#endif
}

}  // namespace

void* HostMemoryResource::do_allocate(
    std::size_t size, [[maybe_unused]] rmm::cuda_stream_view stream, std::size_t alignment
) {
    void* ret = ::operator new(size, std::align_val_t{alignment});
    enable_hugepage_for_region(ret, size);
    return ret;
}

void HostMemoryResource::do_deallocate(
    void* ptr,
    [[maybe_unused]] std::size_t size,
    rmm::cuda_stream_view stream,
    std::size_t alignment
) noexcept {
    stream.synchronize();
    ::operator delete(ptr, std::align_val_t{alignment});
}
}  // namespace rapidsmpf
