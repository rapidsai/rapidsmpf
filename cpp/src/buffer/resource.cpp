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

#include <utility>

#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


AllocToken::~AllocToken() noexcept {
    br->release(*this);
}

std::pair<std::unique_ptr<AllocToken>, std::size_t> BufferResource::reserve(
    MemoryType mem_type, size_t size
) {
    constexpr std::size_t overbooking = 0;
    return {std::make_unique<AllocToken>(mem_type, this, size), overbooking};
}

void BufferResource::release(AllocToken const& token) noexcept {}

std::unique_ptr<Buffer> BufferResource::allocate(
    MemoryType mem_type,
    size_t size,
    rmm::cuda_stream_view stream,
    std::unique_ptr<AllocToken>& token
) {
    std::unique_ptr<Buffer> ret;
    switch (mem_type) {
    case MemoryType::HOST:
        // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
        // std::pmr::vector?
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(size), this}
        );
        break;
    case MemoryType::DEVICE:
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<rmm::device_buffer>(size, stream, device_mr_), this}
        );
        break;
    default:
        RAPIDSMP_FAIL("MemoryType: unknown");
    }
    token->use(mem_type, size);
    return ret;
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
    MemoryType target,
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    std::unique_ptr<AllocToken>& token
) {
    if (target != buffer->mem_type) {
        auto ret = buffer->copy(target, stream);
        token->use(target, ret->size);
        return ret;
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    std::unique_ptr<AllocToken>& token
) {
    return std::move(move(MemoryType::DEVICE, std::move(buffer), stream, token)->device()
    );
}

std::unique_ptr<std::vector<uint8_t>> BufferResource::move_to_host_vector(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    std::unique_ptr<AllocToken>& token
) {
    return std::move(move(MemoryType::HOST, std::move(buffer), stream, token)->host());
}

std::unique_ptr<Buffer> BufferResource::copy(
    MemoryType target,
    std::unique_ptr<Buffer> const& buffer,
    rmm::cuda_stream_view stream,
    std::unique_ptr<AllocToken>& token
) {
    auto ret = buffer->copy(target, stream);
    if (target != buffer->mem_type) {
        token->use(target, ret->size);
    }
    return ret;
}


}  // namespace rapidsmp
