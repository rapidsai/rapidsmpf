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

    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<Buffer> gpu_data
    )
        : metadata{std::move(metadata)}, gpu_data{std::move(gpu_data)} {}

    PackedData(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        BufferResource* br,
        rmm::cuda_stream_view stream,
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

    // /**
    //  * @brief Construct an empty PackedData object.
    //  *
    //  * This constructor initializes both the metadata and GPU data to empty
    //  * buffers.
    //  */
    // PackedData()
    //     : metadata{std::make_unique<std::vector<std::uint8_t>>()},
    //       gpu_data{std::make_unique<rmm::device_buffer>()} {}

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
     */
    [[nodiscard]] bool empty() const {
        return metadata->empty() && gpu_data->size == 0;
    }
};

}  // namespace rapidsmpf
