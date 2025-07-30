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
    std::unique_ptr<Buffer> gpu_data;  ///< The gpu data

    PackedData() = default;

    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<Buffer> gpu_data
    )
        : metadata{std::move(metadata)}, gpu_data{std::move(gpu_data)} {}

    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Buffer::Event> event = nullptr
    )
        : PackedData(std::move(metadata), br->move(std::move(gpu_data), stream, event)) {}

    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        BufferResource* br
    )
        : metadata{std::move(metadata)} {
        auto stream = gpu_data->stream();
        this->gpu_data = br->move(std::move(gpu_data), stream);
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
    PackedData& operator=(PackedData&) = delete;

    /**
     * @brief Check if the packed data is empty.
     *
     * @return True if the packed data is empty, false otherwise.
     *
     * @throw std::invalid_argument if metadata and gpu_data is null and non-null.
     */
    [[nodiscard]] bool empty() const {
        RAPIDSMPF_EXPECTS(
            (!metadata) == (!gpu_data),
            "the metadata and gpu_data pointers cannot be null and non-null",
            std::invalid_argument
        );
        return metadata->empty() && gpu_data->size == 0;
    }
};

}  // namespace rapidsmpf
