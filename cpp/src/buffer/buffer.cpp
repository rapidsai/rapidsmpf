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

Buffer::~Buffer() noexcept {
    if (!is_moved()) {
        br->finalizer(this);
    }
}

Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br)
    : host_buffer_{std::move(host_buffer)},
      mem_type{MemoryType::HOST},
      br{br},
      size{host_buffer_ ? host_buffer_->size() : 0} {
    RAPIDSMP_EXPECTS(host_buffer_ != nullptr, "the host_buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br)
    : device_buffer_{check_null(std::move(device_buffer))},
      mem_type{MemoryType::DEVICE},
      br{br},
      size{device_buffer_->size()} {
    RAPIDSMP_EXPECTS(device_buffer_ != nullptr, "the device_buffer_ cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

bool Buffer::is_moved() const noexcept {
    switch (mem_type) {
    case MemoryType::HOST:
        return host_buffer_ == nullptr;
    case MemoryType::DEVICE:
        return device_buffer_ == nullptr;
    }
    // This cannot happen, `mem_type` is always a member of `MemoryType`.
    return true;
}

void* Buffer::data() {
    switch (mem_type) {
    case MemoryType::HOST:
        return host()->data();
    case MemoryType::DEVICE:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

void const* Buffer::data() const {
    switch (mem_type) {
    case MemoryType::HOST:
        return host()->data();
    case MemoryType::DEVICE:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy(rmm::cuda_stream_view stream) const {
    switch (mem_type) {
    case MemoryType::HOST:
        return std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(*host()), br}
        );
    case MemoryType::DEVICE:
        return std::make_unique<Buffer>(Buffer{
            std::make_unique<rmm::device_buffer>(
                device()->data(), device()->size(), stream, br->device_mr()
            ),
            br
        });
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy(MemoryType target, rmm::cuda_stream_view stream)
    const {
    // Implement the copy between each possible memory types (both directions).
    switch (mem_type) {
    case MemoryType::HOST:
        switch (target) {
        case MemoryType::HOST:  // host -> host
            return copy(stream);
        case MemoryType::DEVICE:  // host -> device
            return std::make_unique<Buffer>(Buffer{
                std::make_unique<rmm::device_buffer>(
                    host()->data(), host()->size(), stream, br->device_mr()
                ),
                br
            });
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    case MemoryType::DEVICE:
        switch (target) {
        case MemoryType::HOST:  // device -> host
            {
                auto ret = std::make_unique<std::vector<uint8_t>>(device()->size());
                RMM_CUDA_TRY(cudaMemcpyAsync(
                    ret->data(),
                    device()->data(),
                    device()->size(),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                return std::make_unique<Buffer>(Buffer{std::move(ret), br});
            }
        case MemoryType::DEVICE:  // device -> device
            return copy(stream);
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

}  // namespace rapidsmp
