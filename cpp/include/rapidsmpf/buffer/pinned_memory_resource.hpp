/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include <cuda/stream_ref>

namespace rapidsmpf {

class PinnedMemoryResource;  // forward declaration

/**
 * @brief Properties for configuring a pinned memory pool. It is aimed to mimic
 * `cuda::experimental::memory_pool_properties`. See for more details:
 * https://nvidia.github.io/cccl/cudax/api/structcuda_1_1experimental_1_1memory__pool__properties.html
 *
 * Currently, this is a placeholder and does not have any effect. It was observed that
 * priming async pools have little effect for performance. See for more details:
 * https://github.com/rapidsai/rmm/issues/1931
 */
struct PinnedPoolProperties {};

/**
 * @brief A pinned host memory pool for stream-ordered allocations/deallocations. This
 * internally uses `cuda::experimental::pinned_memory_pool`. See for more details:
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__pool.html
 */
class PinnedMemoryPool {
    friend class PinnedMemoryResource;

  public:
    /**
     * @brief Constructs a new pinned memory pool.
     *
     * @param numa_id The NUMA node ID to associate with this pool. Default is 0.
     * @param properties Configuration properties for the memory pool.
     */
    PinnedMemoryPool(int numa_id = 0, PinnedPoolProperties properties = {});

    /**
     * @brief Destroys the pinned memory pool.
     *
     * Releases all memory associated with this pool.
     */
    ~PinnedMemoryPool();

    /**
     * @brief Gets the NUMA node ID associated with this pool.
     *
     * @return The NUMA node ID.
     */
    [[nodiscard]] constexpr int numa_id() const noexcept {
        return numa_id_;
    }

    /**
     * @brief Gets the properties used to configure this pool.
     *
     * @return A const reference to the pool properties.
     */
    [[nodiscard]] constexpr PinnedPoolProperties const& properties() const noexcept {
        return properties_;
    }

  private:
    int numa_id_;  ///< The NUMA node ID associated with this pool.
    PinnedPoolProperties properties_;  ///< Configuration properties for this pool.

    struct PinnedMemoryPoolImpl;
    std::unique_ptr<PinnedMemoryPoolImpl> impl_;
};

/**
 * @brief A memory resource that allocates/deallocates pinned host memory from a pinned
 * host memory pool. This internally uses
 * `cuda::experimental::pinned_memory_resource`. See for more details:
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__resource.html
 *
 * This class provides an interface for allocating and deallocating pinned
 * (page-locked) host memory asynchronously using CUDA streams.
 */
class PinnedMemoryResource {
  public:
    /**
     * @brief Constructs a new pinned memory resource.
     *
     * @param pool The pinned memory pool to use for allocations.
     */
    PinnedMemoryResource(PinnedMemoryPool& pool);

    /**
     * @brief Destroys the pinned memory resource.
     *
     * Note: This does not deallocate memory that was allocated through this resource.
     * All allocated memory should be explicitly deallocated before destruction.
     */
    ~PinnedMemoryResource();

    /**
     * @brief Allocates pinned memory asynchronously.
     *
     * @param bytes The number of bytes to allocate.
     * @param stream_ref The CUDA stream to use for the allocation operation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate_async(size_t bytes, const cuda::stream_ref stream_ref);

    /**
     * @brief Deallocates pinned memory asynchronously.
     *
     * @param ptr A pointer to the memory to deallocate.
     * @param stream_ref The CUDA stream to use for the deallocation operation.
     */
    void deallocate_async(void* ptr, const cuda::stream_ref stream_ref);

  private:
    struct PinnedMemoryResourceImpl;
    std::unique_ptr<PinnedMemoryResourceImpl> impl_;
};

/**
 * @brief A buffer that manages pinned host memory.
 *
 * This class provides a convenient interface for managing pinned host memory
 * buffers. It automatically handles allocation and deallocation through a
 * PinnedMemoryResource and supports both empty buffer creation and buffer
 * creation with data copying.
 */
class PinnedHostBuffer {
  public:
    /**
     * @brief Constructs an empty pinned host buffer.
     *
     * @param size The size of the buffer in bytes.
     * @param stream The CUDA stream to use for memory operations.
     * @param mr Shared pointer to the memory resource to use for allocation and
     * deallocation.
     */
    PinnedHostBuffer(
        size_t size, cuda::stream_ref stream, std::shared_ptr<PinnedMemoryResource> mr
    );

    /**
     * @brief Constructs a pinned host buffer and copies data into it.
     *
     * @param src_data Pointer to the source data to copy.
     * @param size The size of the data to copy in bytes.
     * @param stream The CUDA stream to use for memory operations.
     * @param mr Shared pointer to the memory resource to use for allocation and
     * deallocation.
     */
    PinnedHostBuffer(
        void const* src_data,
        size_t size,
        cuda::stream_ref stream,
        std::shared_ptr<PinnedMemoryResource> mr
    );

    /**
     * @brief Constructs a pinned host buffer by copying data from another pinned host
     * buffer on the given stream.
     *
     * @param other The other pinned host buffer to copy from.
     * @param stream The CUDA stream to use for memory operations.
     * @param mr Shared pointer to the memory resource to use for allocation and
     * deallocation.
     */
    PinnedHostBuffer(
        PinnedHostBuffer const& other,
        cuda::stream_ref stream,
        std::shared_ptr<PinnedMemoryResource> mr
    );

    /**
     * @brief Destroys the pinned host buffer and waits for the associated stream to
     * complete.
     */
    ~PinnedHostBuffer() noexcept;

    /**
     * @brief Move constructor.
     *
     * @param other The other pinned host buffer to move from.
     */
    PinnedHostBuffer(PinnedHostBuffer&& other);

    /**
     * @brief Move assignment operator.
     *
     * @param other The other pinned host buffer to move from.
     * @return Moved this.
     */
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other);

    // copy constructor and assignment operator are deleted. Use stream variant instead.
    PinnedHostBuffer(const PinnedHostBuffer& other) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer& other) = delete;

    /**
     * @brief Deallocates the buffer memory asynchronously.
     *
     * This method can be called to explicitly deallocate the buffer memory
     * before the destructor is called. After calling this method, the buffer
     * data pointer will be set to nullptr.
     */
    void deallocate_async() noexcept;

    /**
     * @brief Gets a const pointer to the buffer data.
     *
     * @return A const pointer to the buffer data, or nullptr if not allocated.
     */
    [[nodiscard]] constexpr std::byte const* data() const noexcept {
        return data_;
    }

    /**
     * @brief Gets a pointer to the buffer data.
     *
     * @return A pointer to the buffer data, or nullptr if not allocated.
     */
    constexpr std::byte* data() noexcept {
        return data_;
    }

    /**
     * @brief Gets the size of the buffer.
     *
     * @return The size of the buffer in bytes.
     */
    [[nodiscard]] constexpr size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Gets the CUDA stream associated with this buffer.
     *
     * @return The CUDA stream reference.
     */
    [[nodiscard]] constexpr cuda::stream_ref stream_ref() const noexcept {
        return stream_ref_;
    }

    /**
     * @brief Sets the CUDA stream for this buffer.
     *
     * @param stream The new CUDA stream to use for memory operations.
     */
    constexpr void set_stream(cuda::stream_ref stream) noexcept {
        stream_ref_ = stream;
    }

    std::byte* data_ = nullptr;  ///< Pointer to the allocated buffer data.
    size_t size_;  ///< Size of the buffer in bytes.
    cuda::stream_ref stream_ref_;  ///< CUDA stream used for memory operations.
    std::shared_ptr<PinnedMemoryResource>
        mr_;  ///< Shared pointer to the memory resource used for allocation/deallocation.
};

}  // namespace rapidsmpf
