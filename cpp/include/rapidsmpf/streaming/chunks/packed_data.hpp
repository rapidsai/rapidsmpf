/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/buffer/packed_data.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of `PackedData`.
 */
struct PackedDataChunk {
    /**
     * @brief Packed data payload.
     */
    PackedData data;
};

}  // namespace rapidsmpf::streaming
