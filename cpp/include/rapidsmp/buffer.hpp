/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <memory>
#include <vector>

#include <rmm/device_buffer.hpp>

#include <rapidsmp/error.hpp>

namespace rapidsmp {

/**
 * @brief Buffer representing device or host memory.
 *
 * The memory type (host or device) is constant and cannot change during the
 * object's lifetime, which simplify multi-threading.
 */
class Buffer {
  public:
    /// @brief Enum representing the type of memory.
    enum class MemType {
        host,  ///< Host memory
        device  ///< Device memory
    };

    /**
     * @brief Construct a Buffer from host memory.
     *
     * @param host_buffer A unique pointer to a vector containing host memory.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param mr Memory resource for device memory allocation.
     *
     * @throws std::invalid_argument if `host_buffer` is null.
     */
    Buffer(
        std::unique_ptr<std::vector<uint8_t>> host_buffer,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    );

    /**
     * @brief Construct a Buffer from device memory.
     *
     * @param device_buffer A unique pointer to a device buffer.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param mr Memory resource for device memory allocation.
     *
     * @throws std::invalid_argument if `device_buffer` is null.
     * @throws std::invalid_argument if `stream` or `mr` isn't the same used by
     * `device_buffer`.
     */
    Buffer(
        std::unique_ptr<rmm::device_buffer> device_buffer,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    );

    /**
     * @brief Construct a Buffer from device memory.
     *
     * The CUDA stream and RMM memory resource are inferred from `device_buffer`.
     *
     * @param device_buffer A unique pointer to a device buffer.
     *
     * @throws std::invalid_argument if `device_buffer` is null.
     */
    Buffer(std::unique_ptr<rmm::device_buffer> device_buffer);

    /**
     * @brief Access the underlying host memory buffer.
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>>& host() {
        RAPIDSMP_EXPECTS(mem_type == MemType::host, "buffer not is host memory");
        return host_buffer_;
    }

    /**
     * @brief Access the underlying host memory buffer (const).
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> const& host() const {
        RAPIDSMP_EXPECTS(mem_type == MemType::host, "buffer not is host memory");
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
        RAPIDSMP_EXPECTS(mem_type == MemType::device, "buffer not in device memory");
        return device_buffer_;
    }

    /**
     * @brief Access the underlying device memory buffer (const).
     *
     * @return A const reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] std::unique_ptr<rmm::device_buffer> const& device() const {
        RAPIDSMP_EXPECTS(mem_type == MemType::device, "buffer not in device memory");
        return device_buffer_;
    }

    /**
     * @brief Create a copy of this buffer in device memory.
     *
     * @return A unique pointer to a new Buffer containing the copied data in device
     * memory.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy_to_device() const;

    /**
     * @brief Create a copy of this buffer in host memory.
     *
     * @return A unique pointer to a new Buffer containing the copied data in host memory.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy_to_host() const;

    /**
     * @brief Move a buffer to the specified memory type.
     *
     * Copies the buffer if moving between memory types.
     *
     * @param buffer The buffer to move.
     * @param target The target memory type.
     * @return A unique pointer to the moved Buffer.
     */
    [[nodiscard]] static std::unique_ptr<Buffer> move(
        std::unique_ptr<Buffer>&& buffer, MemType target
    );

  private:
    /// @brief The underlying host memory buffer (if applicable).
    std::unique_ptr<std::vector<uint8_t>> host_buffer_;
    /// @brief The underlying device memory buffer (if applicable).
    std::unique_ptr<rmm::device_buffer> device_buffer_;

  public:
    MemType const mem_type;  ///< Memory type.
    rmm::cuda_stream_view const stream;  ///< The CUDA stream used.
    rmm::device_async_resource_ref const mr;  ///< The RMM memory resource used.
    size_t const size;  ///< The size of the buffer in bytes.
};


}  // namespace rapidsmp
