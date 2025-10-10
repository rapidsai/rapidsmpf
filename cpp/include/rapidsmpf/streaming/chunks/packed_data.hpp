/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <rapidsmpf/buffer/packed_data.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Chunk of `PackedData` with sequence number.
 */
struct PackedDataChunk {
    /**
     * @brief Sequence number used to preserve chunk ordering.
     */
    std::uint64_t sequence_number;

    /**
     * @brief Packed data payload.
     */
    PackedData data;
};

}  // namespace rapidsmpf::streaming
