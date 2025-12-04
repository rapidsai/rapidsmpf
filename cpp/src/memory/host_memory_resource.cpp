/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/memory/host_memory_resource.hpp>

namespace rapidsmpf {

void* HostMemoryResource::do_allocate(
    std::size_t size, [[maybe_unused]] rmm::cuda_stream_view stream, std::size_t alignment
) {
    return ::operator new(size, std::align_val_t{alignment});
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
