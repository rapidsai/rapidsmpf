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
 * @brief
 *
 * The memory type is constant so don't worry about the type changing underneath you
 */
class Buffer {
  public:
    enum class MemType {
        host,
        device
    };
    MemType const mem_type;
    rmm::cuda_stream_view const stream;
    rmm::device_async_resource_ref const mr;
    size_t const size;

    Buffer(
        std::unique_ptr<std::vector<uint8_t>> host_buffer,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    )
        : mem_type{MemType::host},
          stream{stream},
          mr{mr},
          size{host_buffer ? host_buffer->size() : 0},
          host_buffer_{std::move(host_buffer)} {}

    Buffer(
        std::unique_ptr<rmm::device_buffer> device_buffer,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    )
        : mem_type{MemType::device},
          stream{stream},
          mr{mr},
          size{device_buffer ? device_buffer->size() : 0},
          device_buffer_{std::move(device_buffer)} {
        RAPIDSMP_EXPECTS(
            device_buffer_->stream() == stream, "the CUDA streams doesn't match"
        );
        RAPIDSMP_EXPECTS(
            device_buffer_->memory_resource() == mr,
            "the RMM memory resources doesn't match"
        );
    }

    Buffer(std::unique_ptr<rmm::device_buffer> device_buffer)
        : mem_type{MemType::device},
          stream{device_buffer->stream()},
          mr{device_buffer->memory_resource()},
          size{device_buffer ? device_buffer->size() : 0},
          device_buffer_{std::move(device_buffer)} {}

    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>>& host() {
        RAPIDSMP_EXPECTS(mem_type == MemType::host, "buffer not is host memory");
        return host_buffer_;
    }

    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> const& host() const {
        RAPIDSMP_EXPECTS(mem_type == MemType::host, "buffer not is host memory");
        return host_buffer_;
    }

    [[nodiscard]] std::unique_ptr<rmm::device_buffer>& device() {
        RAPIDSMP_EXPECTS(mem_type == MemType::device, "buffer not in device memory");
        return device_buffer_;
    }

    [[nodiscard]] std::unique_ptr<rmm::device_buffer> const& device() const {
        RAPIDSMP_EXPECTS(mem_type == MemType::device, "buffer not in device memory");
        return device_buffer_;
    }

    [[nodiscard]] std::unique_ptr<Buffer> copy_to_device(

    ) const;

    [[nodiscard]] std::unique_ptr<Buffer> copy_to_host(

    ) const;

    [[nodiscard]] static std::unique_ptr<Buffer> move(
        std::unique_ptr<Buffer>&& buffer, MemType target
    );


  private:
    std::unique_ptr<std::vector<uint8_t>> host_buffer_;
    std::unique_ptr<rmm::device_buffer> device_buffer_;
};


}  // namespace rapidsmp
