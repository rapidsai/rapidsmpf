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

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {

namespace {
// Check that `ptr` isn't null.
template <typename T>
[[nodiscard]] std::unique_ptr<T> check_null(std::unique_ptr<T> ptr) {
    RAPIDSMP_EXPECTS(ptr, "unique pointer cannot be null", std::invalid_argument);
    return std::move(ptr);
}
}  // namespace

Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br)
    : host_buffer_{std::move(host_buffer)},
      mem_type{MemoryType::host},
      br{br},
      size{host_buffer_ ? host_buffer_->size() : 0} {
    RAPIDSMP_EXPECTS(host_buffer_ != nullptr, "the host_buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br)
    : device_buffer_{check_null(std::move(device_buffer))},
      mem_type{MemoryType::device},
      br{br},
      size{device_buffer_->size()} {
    RAPIDSMP_EXPECTS(device_buffer_ != nullptr, "the device_buffer_ cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
    RAPIDSMP_EXPECTS(
        device_buffer_->memory_resource() == br->device_mr(),
        "the RMM memory resources doesn't match"
    );
}

bool Buffer::is_moved() const noexcept {
    switch (mem_type) {
    case MemoryType::host:
        return host_buffer_ == nullptr;
    case MemoryType::device:
        return device_buffer_ == nullptr;
    }
    // This cannot happen, `mem_type` is always a member of `MemoryType`.
    return true;
}

void* Buffer::data() {
    switch (mem_type) {
    case MemoryType::host:
        return host()->data();
    case MemoryType::device:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

void const* Buffer::data() const {
    switch (mem_type) {
    case MemoryType::host:
        return host()->data();
    case MemoryType::device:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy_to_device(rmm::cuda_stream_view stream) const {
    std::unique_ptr<rmm::device_buffer> ret;
    if (mem_type == MemoryType::device) {
        ret = std::make_unique<rmm::device_buffer>(
            device()->data(), device()->size(), stream, br->device_mr()
        );
    } else {
        ret = std::make_unique<rmm::device_buffer>(
            host()->data(), host()->size(), stream, br->device_mr()
        );
    }
    return std::make_unique<Buffer>(Buffer{std::move(ret), br});
}

std::unique_ptr<Buffer> Buffer::copy_to_host(rmm::cuda_stream_view stream) const {
    std::unique_ptr<std::vector<uint8_t>> ret;
    if (mem_type == MemoryType::host) {
        ret = std::make_unique<std::vector<uint8_t>>(*host());
    } else {
        ret = std::make_unique<std::vector<uint8_t>>(device()->size());
        RMM_CUDA_TRY(cudaMemcpyAsync(
            ret->data(),
            device()->data(),
            device()->size(),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }
    return std::make_unique<Buffer>(Buffer{std::move(ret), br});
}


}  // namespace rapidsmp
