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

/**
 * @brief Callback function to resolves the memory type based on a size input.
 *
 * Used to determine whether memory should be allocated on host or device
 * given the buffer size.
 */
using MemoryTypeResolver = std::function<MemoryType(std::size_t)>;

namespace memory_type_resolver {

/**
 * @brief A constant memory type resolver.
 *
 * Always returns the specified memory type, regardless of the allocation size.
 */
struct constant {
    /**
     * @brief Constructs a constant memory type resolver.
     *
     * @param mem_type The memory type to always resolve to.
     */
    constant(MemoryType mem_type) : mem_type{mem_type} {}

    /**
     * @brief Resolve to the constant memory type.
     *
     * @param size Input size (ignored).
     * @return The constant memory type.
     */
    MemoryType operator()(std::size_t size) const {
        return mem_type;
    }

    MemoryType const mem_type;  ///< The constant memory type to resolve.
};

}  // namespace memory_type_resolver

/**
 * @brief Base class for managing buffer resources.
 *
 * This class handles memory allocation and transfers between different memory types
 * (e.g., host and device). All memory operations in rapidsmp, such as those performed
 * by the Shuffler, rely on a buffer resource for memory management.
 *
 * This base class use the provided `MemoryTypeResolver` callback function to determine
 * the memory type (host or device) of an allocation.
 *
 * An alternative allocation strategic can be implemented in a derived class by overriding
 * one or more of the virtual methods.
 *
 * @note Similar to RMM's memory resource, the `BufferResource` instance must outlive all
 * allocated buffers.
 */
class BufferResource {
  public:
    /**
     * @brief Constructs a buffer resource that uses a memory resolver to decide the
     * memory type of each new allocation.
     *
     * @param device_mr Reference to the RMM device memory resource used for all device
     * allocations, which must outlive `BufferResource` and all the created buffers.
     * @param resolver Memory type resolver, which is a callback function that takes an
     * allocation size and returns a memory type. The default resolver always returns
     * device memory.
     */
    BufferResource(
        rmm::device_async_resource_ref device_mr,
        MemoryTypeResolver resolver = memory_type_resolver::constant(MemoryType::device)
    )
        : device_mr_{device_mr}, resolver_{std::move(resolver)} {}

    virtual ~BufferResource() noexcept = default;

    /**
     * @brief Get the RMM device memory resource.
     *
     * @return Reference to the RMM resource used for device allocations.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() const noexcept {
        return device_mr_;
    }

    /**
     * @brief Allocate a buffer of the specified memory type.
     *
     * @param mem_type The target memory type (host or device).
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @return A unique pointer to the allocated Buffer.
     */
    virtual std::unique_ptr<Buffer> allocate(
        MemoryType mem_type, size_t size, rmm::cuda_stream_view stream
    );

    /**
     * @brief Allocate a buffer based on the memory type resolver.
     *
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @return A unique pointer to the allocated Buffer.
     */
    virtual std::unique_ptr<Buffer> allocate(size_t size, rmm::cuda_stream_view stream) {
        return allocate(resolver_(size), size, stream);
    }

    /**
     * @brief Move host vector data into a Buffer.
     *
     * @param data A unique pointer to the vector containing host data.
     * @param stream CUDA stream for any necessary operations.
     * @return A unique pointer to the resulting Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move device buffer data into a Buffer.
     *
     * @param data A unique pointer to the device buffer.
     * @param stream CUDA stream for any necessary operations.
     * @return A unique pointer to the resulting Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move a Buffer to the specified memory type.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param target The target memory type.
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the moved Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        MemoryType target, std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move a Buffer to a device buffer.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the resulting device buffer.
     */
    virtual std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move a Buffer to a host vector.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the resulting host vector.
     */
    virtual std::unique_ptr<std::vector<uint8_t>> move_to_host_vector(
        std::unique_ptr<Buffer> buffer, rmm::cuda_stream_view stream
    );

    /**
     * @brief Create a copy of a Buffer in the specified memory type.
     *
     * Unlike `move()`, this always performs a copy operation.
     *
     * @param target The target memory type.
     * @param buffer The buffer to copy.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the new Buffer.
     */
    virtual std::unique_ptr<Buffer> copy(
        MemoryType target,
        std::unique_ptr<Buffer> const& buffer,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief This finalizer is called when a buffer is being destructured.
     *
     * This base implementation does nothing.
     *
     * @note This is only called on initialized buffers i.e. only if
     * `buffer->is_moved() == false`.
     *
     * @param buffer The buffer being destructured.
     */
    virtual void finalizer(Buffer* const buffer) noexcept {}

  protected:
    rmm::device_async_resource_ref device_mr_;  ///< RMM device memory resource reference.
    MemoryTypeResolver resolver_;  ///< Function to resolve memory type.
};

}  // namespace rapidsmp
