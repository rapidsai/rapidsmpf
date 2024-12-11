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
#include <stdexcept>

#include <rapidsmp/buffer.hpp>

namespace rapidsmp {

namespace {
// Check that `ptr` isn't null.
template <typename T>
[[nodiscard]] std::unique_ptr<T> check_null(std::unique_ptr<T> ptr) {
    RAPIDSMP_EXPECTS(ptr, "unique pointer cannot be null", std::invalid_argument);
    return std::move(ptr);
}
}  // namespace

Buffer::Buffer(
    std::unique_ptr<std::vector<uint8_t>> host_buffer,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
)
    : host_buffer_{std::move(host_buffer)},
      mem_type{MemType::host},
      stream{stream},
      mr{mr},
      size{host_buffer_ ? host_buffer_->size() : 0} {}

Buffer::Buffer(
    std::unique_ptr<rmm::device_buffer> device_buffer,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
)
    : device_buffer_{check_null(std::move(device_buffer))},
      mem_type{MemType::device},
      stream{stream},
      mr{mr},
      size{device_buffer_->size()} {
    RAPIDSMP_EXPECTS(
        device_buffer_->stream() == stream,
        "the CUDA streams doesn't match",
        std::invalid_argument
    );
    RAPIDSMP_EXPECTS(
        device_buffer_->memory_resource() == mr, "the RMM memory resources doesn't match"
    );
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer)
    : device_buffer_{check_null(std::move(device_buffer))},
      mem_type{MemType::device},
      stream{device_buffer_->stream()},
      mr{device_buffer_->memory_resource()},
      size{device_buffer_->size()} {}

std::unique_ptr<Buffer> Buffer::copy_to_device() const {
    std::unique_ptr<rmm::device_buffer> ret;
    if (mem_type == MemType::device) {
        ret = std::make_unique<rmm::device_buffer>(
            device()->data(), device()->size(), stream, mr
        );
    } else {
        ret = std::make_unique<rmm::device_buffer>(
            host()->data(), host()->size(), stream, mr
        );
    }
    return std::make_unique<Buffer>(std::move(ret), stream, mr);
}

std::unique_ptr<Buffer> Buffer::copy_to_host() const {
    std::unique_ptr<std::vector<uint8_t>> ret;
    if (mem_type == MemType::host) {
        ret = std::make_unique<std::vector<uint8_t>>(*host());
    } else {
        ret = std::make_unique<std::vector<uint8_t>>(device()->size());
        RMM_CUDA_TRY(cudaMemcpyAsync(
            ret->data(), device()->data(), device()->size(), cudaMemcpyDeviceToHost
        ));
    }
    return std::make_unique<Buffer>(std::move(ret), stream, mr);
}

std::unique_ptr<Buffer> Buffer::move(std::unique_ptr<Buffer>&& buffer, MemType target) {
    if (target != buffer->mem_type) {
        switch (buffer->mem_type) {
        case MemType::host:
            return buffer->copy_to_device();
        case MemType::device:
            return buffer->copy_to_host();
        }
    }
    return std::move(buffer);
}


}  // namespace rapidsmp
