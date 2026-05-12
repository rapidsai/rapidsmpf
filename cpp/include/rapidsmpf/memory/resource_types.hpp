/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/memory_resource>

namespace rapidsmpf {

/// @brief Owning type-erased device memory resource.
using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

/// @brief Owning type-erased host- and device-accessible memory resource.
using any_host_device_resource =
    cuda::mr::any_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>;

}  // namespace rapidsmpf
