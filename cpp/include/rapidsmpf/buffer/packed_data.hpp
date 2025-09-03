/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rmm/device_buffer.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @brief Bag of bytes with metadata suitable for sending over the wire.
 *
 * Contains arbitrary gpu data and host side metadata indicating how
 * the data should be interpreted.
 */
struct PackedData {
    std::unique_ptr<std::vector<std::uint8_t>> metadata;  ///< The metadata
    std::unique_ptr<Buffer> data;  ///< The gpu data

    /**
     * @brief Construct from metadata and gpu data, taking ownership.
     *
     * @param metadata Host-side metadata describing the GPU data.
     * @param data Pointer to GPU data.
     */
    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata, std::unique_ptr<Buffer> data
    )
        : metadata{std::move(metadata)}, data{std::move(data)} {
        RAPIDSMPF_EXPECTS(
            this->metadata != nullptr, "the metadata pointer cannot be null"
        );
        RAPIDSMPF_EXPECTS(this->data != nullptr, "the gpu data pointer cannot be null");
        RAPIDSMPF_EXPECTS(
            (this->metadata->size() > 0 || this->data->size == 0),
            "Empty Metadata and non-empty GPU data is not allowed"
        );
    }

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
    PackedData& operator=(PackedData const&) = delete;

    /**
     * @brief Check if the packed data is empty.
     *
     * @return True if the packed data is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const {
        return metadata->empty() && data->size == 0;
    }
};

}  // namespace rapidsmpf
