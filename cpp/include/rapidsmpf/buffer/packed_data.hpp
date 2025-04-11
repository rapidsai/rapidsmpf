/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rmm/device_buffer.hpp>

namespace rapidsmpf {

/**
 * @brief Bag of bytes with metadata suitable for sending over the wire.
 *
 * Contains arbitrary gpu data and host side metadata indicating how
 * the data should be interpreted.
 */
struct PackedData {
    std::unique_ptr<std::vector<std::uint8_t>> metadata{nullptr};  ///< The metadata
    std::unique_ptr<rmm::device_buffer> gpu_data{nullptr};  ///< The gpu data

    PackedData() : metadata{nullptr}, gpu_data{nullptr} {}

    /**
     * @brief Construct packed data from metadata and gpu data, taking ownership.
     *
     * @param meta The metadata
     * @param data The gpu data
     */
    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>>&& meta,
        std::unique_ptr<rmm::device_buffer>&& data
    )
        : metadata{std::move(meta)}, gpu_data{std::move(data)} {}

    ~PackedData() = default;
    /// @brief PackedData is moveable
    PackedData(PackedData&&) = default;
    /**
     * @brief Move assignment
     *
     * @returns Moved this.
     */
    PackedData& operator=(PackedData&&) = default;
    PackedData(PackedData const&) = delete;
    PackedData& operator=(PackedData&) = delete;
};

}  // namespace rapidsmpf
