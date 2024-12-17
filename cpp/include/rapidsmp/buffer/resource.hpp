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


#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

class BufferResource {
  public:
    BufferResource(rmm::device_async_resource_ref device_mr) : device_mr_{device_mr} {}

    virtual ~BufferResource() noexcept = default;

    /**
     * @brief The RMM resource used to allocate and deallocate device memory
     *
     * @return Reference to the RMM resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() const noexcept {
        return device_mr_;
    }

    virtual std::unique_ptr<Buffer> allocate(
        MemoryType mem_type, size_t size, rmm::cuda_stream_view stream
    ) {
        switch (mem_type) {
        case MemoryType::host:
            // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
            // std::pmr::vector?
            return std::make_unique<Buffer>(
                std::make_unique<std::vector<uint8_t>>(size), stream, device_mr_
            );
        case MemoryType::device:
            return std::make_unique<Buffer>(
                std::make_unique<rmm::device_buffer>(size, stream, device_mr_),
                stream,
                device_mr_
            );
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    }

    virtual std::unique_ptr<Buffer> allocate(size_t size, rmm::cuda_stream_view stream) {
        return allocate(MemoryType::device, size, stream);
    }

    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
    ) {
        return std::make_unique<Buffer>(std::move(data), stream, device_mr_);
    }

    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    ) {
        return std::make_unique<Buffer>(std::move(data), stream, device_mr_);
    }

    /**
     * @brief Move a buffer to the specified memory type.
     *
     * Copies the buffer if moving between memory types.
     *
     * @param buffer The buffer to move.
     * @param target The target memory type.
     * @return A unique pointer to the moved Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<Buffer> buffer, MemoryType target
    ) {
        if (target != buffer->mem_type) {
            switch (buffer->mem_type) {
            case MemoryType::host:
                return buffer->copy_to_device();
            case MemoryType::device:
                return buffer->copy_to_host();
            }
            RAPIDSMP_FAIL("MemoryType: unknown");
        }
        return buffer;
    }

  protected:
    rmm::device_async_resource_ref device_mr_;
};


}  // namespace rapidsmp
