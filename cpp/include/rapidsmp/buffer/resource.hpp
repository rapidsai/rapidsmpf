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

#include <functional>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {
using MemoryTypeResolver = std::function<MemoryType(std::size_t)>;

namespace memory_type_resolver {

struct constant {
    constant(MemoryType mem_type) : mem_type{mem_type} {}

    MemoryType operator()(std::size_t) const {
        std::cout << "memory_type_resolver: ";
        if (mem_type == MemoryType::device) {
            std::cout << "device";
        } else {
            std::cout << "host";
        }
        std::cout << std::endl;
        return mem_type;
    }

    MemoryType const mem_type;
};

}  // namespace memory_type_resolver

class BufferResource {
  public:
    BufferResource(
        rmm::device_async_resource_ref mr,
        MemoryTypeResolver resolver = memory_type_resolver::constant(MemoryType::device)
    )
        : device_mr_{mr}, resolver_{std::move(resolver)} {}

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
                Buffer{std::make_unique<std::vector<uint8_t>>(size), this}
            );
        case MemoryType::device:
            return std::make_unique<Buffer>(Buffer{
                std::make_unique<rmm::device_buffer>(size, stream, device_mr_), this
            });
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    }

    virtual std::unique_ptr<Buffer> allocate(size_t size, rmm::cuda_stream_view stream) {
        return allocate(resolver_(size), size, stream);
    }

    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
    ) {
        return std::make_unique<Buffer>(Buffer{std::move(data), this});
    }

    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    ) {
        return std::make_unique<Buffer>(Buffer{std::move(data), this});
    }

    /**
     * @brief Move a buffer to the specified memory type.
     *
     * Copies the buffer if moving between memory types.
     *
     * @param target The target memory type.
     * @param buffer The buffer to move.
     * @return A unique pointer to the moved Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        MemoryType target, std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    ) {
        if (target != buffer->mem_type) {
            switch (buffer->mem_type) {
            case MemoryType::host:
                return buffer->copy_to_device(stream);
            case MemoryType::device:
                return buffer->copy_to_host(stream);
            }
            RAPIDSMP_FAIL("MemoryType: unknown");
        }
        return buffer;
    }

    virtual std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    ) {
        return std::move(move(MemoryType::device, std::move(buffer), stream)->device());
    }

    virtual std::unique_ptr<std::vector<uint8_t>> move_to_host_vector(
        std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    ) {
        return std::move(move(MemoryType::host, std::move(buffer), stream)->host());
    }

    /**
     * @brief Create a new copy of a buffer in the specified memory type.
     *
     * As opposed to `move()`, this always copy data.
     *
     * @param target The target memory type.
     * @param buffer The buffer to copy.
     * @return A unique pointer to the new Buffer.
     */
    virtual std::unique_ptr<Buffer> copy(
        MemoryType target,
        std::unique_ptr<Buffer> const& buffer,
        rmm::cuda_stream_view stream
    ) {
        switch (buffer->mem_type) {
        case MemoryType::host:
            return buffer->copy_to_device(stream);
        case MemoryType::device:
            return buffer->copy_to_host(stream);
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    }

  protected:
    rmm::device_async_resource_ref device_mr_;
    MemoryTypeResolver resolver_;
};


}  // namespace rapidsmp
