/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <array>
#include <memory>
#include <vector>

#include <rmm/device_buffer.hpp>

#include <rapidsmp/error.hpp>

namespace rapidsmp {

class BufferResource;

/// @brief Enum representing the type of memory.
enum class MemoryType : int {
    DEVICE,  ///< Device memory
    HOST  ///< Host memory
};

/// @brief Array of all the different memory types.
constexpr std::array<MemoryType, 2> MEMORY_TYPES{{MemoryType::DEVICE, MemoryType::HOST}};

/**
 * @brief Buffer representing device or host memory.
 *
 * @note The constructors are private, use `BufferResource` to construct buffers.
 * @note The memory type (e.g., host or device) is constant and cannot change during
 * the buffer's lifetime.
 */
class Buffer {
    friend class BufferResource;

  public:
    /**
     * @brief Check if the buffer has been moved and is now uninitialized.
     *
     * @return Returns true iff the buffer has been moved and should not be accessed.
     */
    [[nodiscard]] bool is_moved() const noexcept;

    /**
     * @brief Access the underlying host memory buffer (const).
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> const& host() const {
        RAPIDSMP_EXPECTS(mem_type == MemoryType::HOST, "buffer is not host memory");
        RAPIDSMP_EXPECTS(host_buffer_, "pointer is null, has the buffer been moved?");
        return host_buffer_;
    }

    /**
     * @brief Access the underlying device memory buffer (const).
     *
     * @return A const reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] std::unique_ptr<rmm::device_buffer> const& device() const {
        RAPIDSMP_EXPECTS(mem_type == MemoryType::DEVICE, "buffer not in device memory");
        RAPIDSMP_EXPECTS(device_buffer_, "pointer is null, has the buffer been moved?");
        return device_buffer_;
    }

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] void* data();

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A const pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] void const* data() const;

    /// @brief Buffer has a move ctor but no copy or assign operator.
    Buffer(Buffer&&) = default;
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer& o) = delete;
    Buffer& operator=(Buffer&& o) = delete;

  private:
    /**
     * @brief Construct a Buffer from host memory.
     *
     * @param host_buffer A unique pointer to a vector containing host memory.
     * @param br Buffer resource for memory allocation.
     *
     * @throws std::invalid_argument if `host_buffer` is null.
     */
    Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br);

    /**
     * @brief Construct a Buffer from device memory.
     *
     * @param device_buffer A unique pointer to a device buffer.
     * @param br Buffer resource for memory allocation.
     *
     * @throws std::invalid_argument if `device_buffer` is null.
     * @throws std::invalid_argument if `stream` or `br->mr` isn't the same used by
     * `device_buffer`.
     */
    Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br);

    /**
     * @brief Access the underlying host memory buffer.
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>>& host() {
        RAPIDSMP_EXPECTS(mem_type == MemoryType::HOST, "buffer is not host memory");
        RAPIDSMP_EXPECTS(host_buffer_, "pointer is null, has the buffer been moved?");
        return host_buffer_;
    }

    /**
     * @brief Access the underlying device memory buffer.
     *
     * @return A reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] std::unique_ptr<rmm::device_buffer>& device() {
        RAPIDSMP_EXPECTS(mem_type == MemoryType::DEVICE, "buffer not in device memory");
        RAPIDSMP_EXPECTS(device_buffer_, "pointer is null, has the buffer been moved?");
        return device_buffer_;
    }

    /**
     * @brief Create a copy of this buffer using the same memory type.
     *
     * @param stream CUDA stream used for device memory operations.
     * @return A unique pointer to a new Buffer containing the copied data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy(rmm::cuda_stream_view stream) const;

    /**
     * @brief Create a copy of this buffer using the specified memory type.
     *
     * @param target The target memory type.
     * @param stream CUDA stream used for device memory operations.
     * @return A unique pointer to a new Buffer containing the copied data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy(
        MemoryType target, rmm::cuda_stream_view stream
    ) const;

  private:
    /// @brief The underlying host memory buffer (if applicable).
    std::unique_ptr<std::vector<uint8_t>> host_buffer_;
    /// @brief The underlying device memory buffer (if applicable).
    std::unique_ptr<rmm::device_buffer> device_buffer_;

  public:
    MemoryType const mem_type;  ///< Memory type.
    BufferResource* const br;  ///< The buffer resource used.
    std::size_t const size;  ///< The size of the buffer in bytes.
};

}  // namespace rapidsmp
