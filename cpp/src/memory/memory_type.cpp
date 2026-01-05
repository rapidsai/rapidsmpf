/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {


MemoryType ptr_to_memory_type(void* ptr) {
    cudaPointerAttributes attrs;
    RAPIDSMPF_CUDA_TRY(cudaPointerGetAttributes(&attrs, ptr));

    switch (attrs.type) {
    case cudaMemoryTypeUnregistered:
        return MemoryType::HOST;
    case cudaMemoryTypeHost:
        return MemoryType::PINNED_HOST;
    case cudaMemoryTypeDevice:
        return MemoryType::DEVICE;
    case cudaMemoryTypeManaged:
        break;
    }

    RAPIDSMPF_FAIL("Unknown memory type", std::runtime_error);
}
}  // namespace rapidsmpf
