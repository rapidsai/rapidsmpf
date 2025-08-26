/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <rmm/cuda_stream_view.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Common base for streaming chunks.
 *
 * Holds the per-chunk sequence number (for ordering) and the CUDA stream
 * on which the chunk's resources were created.
 *
 */
class BaseChunk {
  public:
    /**
     * @brief Construct a base chunk.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param stream CUDA stream associated with the chunk's resources.
     */
    constexpr BaseChunk(
        std::uint64_t sequence_number, rmm::cuda_stream_view stream
    ) noexcept
        : _sequence_number{sequence_number}, _stream{stream} {}

    /**
     * @brief Virtual destructor.
     */
    virtual ~BaseChunk() = default;

    /**
     * @brief move constructor
     */
    BaseChunk(BaseChunk&&) noexcept = default;

    /**
     * @brief move assignment operator
     * @return this chunk.
     */
    BaseChunk& operator=(BaseChunk&&) noexcept = default;

    BaseChunk(BaseChunk const&) = delete;
    BaseChunk& operator=(BaseChunk const&) = delete;

    /**
     * @brief Sequence number used to preserve chunk ordering.
     *
     * @return The sequence number.
     */
    [[nodiscard]] constexpr std::uint64_t sequence_number() const noexcept {
        return _sequence_number;
    }

    /**
     * @brief The CUDA stream on which this chunk was created.
     *
     * @return The associated rmm::cuda_stream_view.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept {
        return _stream;
    }

  private:
    std::uint64_t _sequence_number;
    rmm::cuda_stream_view _stream;
};

}  // namespace rapidsmpf::streaming
