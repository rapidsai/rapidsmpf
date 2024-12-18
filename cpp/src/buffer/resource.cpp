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

#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


std::unique_ptr<Buffer> BufferResource::allocate(
    MemoryType mem_type, size_t size, rmm::cuda_stream_view stream
) {
    switch (mem_type) {
    case MemoryType::HOST:
        // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
        // std::pmr::vector?
        return std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(size), this}
        );
    case MemoryType::DEVICE:
        return std::make_unique<Buffer>(
            Buffer{std::make_unique<rmm::device_buffer>(size, stream, device_mr_), this}
        );
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    MemoryType target, std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
) {
    if (target != buffer->mem_type) {
        return buffer->copy(target, stream);
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
) {
    return std::move(move(MemoryType::DEVICE, std::move(buffer), stream)->device());
}

std::unique_ptr<std::vector<uint8_t>> BufferResource::move_to_host_vector(
    std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
) {
    return std::move(move(MemoryType::HOST, std::move(buffer), stream)->host());
}

std::unique_ptr<Buffer> BufferResource::copy(
    MemoryType target, std::unique_ptr<Buffer> const& buffer, rmm::cuda_stream_view stream
) {
    return buffer->copy(target, stream);
}
}  // namespace rapidsmp
