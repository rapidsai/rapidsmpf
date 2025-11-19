/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>


/// @brief The minimum CUDA version required for PinnedMemoryResource.
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION 12060
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR \
    RAPIDSMPF_STRINGIFY(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)

namespace rapidsmpf {

/**
 * @brief Checks if the PinnedMemoryResource is supported for the current CUDA version.
 *
 * Requires rapidsmpf to be build with cuda>=12.6.
 *
 * @note The driver version check is cached and only performed once.
 */
inline bool is_pinned_memory_resources_supported() {
#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)
    static const bool supported = [] {
        int driver_version = 0;
        RAPIDSMPF_CUDA_TRY(cudaDriverGetVersion(&driver_version));
        return driver_version >= RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION;
    }();
    return supported;
#else
    return false;
#endif
}

class PinnedMemoryResource;  // forward declaration

/**
 * @brief Properties for configuring a pinned memory pool. It is aimed to mimic
 * `cuda::experimental::memory_pool_properties`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/structcuda_1_1experimental_1_1memory__pool__properties.html
 *
 * Currently, this is a placeholder and does not have any effect. It was observed that
 * priming async pools have little effect for performance.
 *
 * @sa https://github.com/rapidsai/rmm/issues/1931
 */
struct PinnedPoolProperties {};

/**
 * @brief A pinned host memory pool for stream-ordered allocations/deallocations. This
 * internally uses `cuda::experimental::pinned_memory_pool`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__pool.html
 */
class PinnedMemoryPool {
    friend class PinnedMemoryResource;

  public:
    /**
     * @brief Constructs a new pinned memory pool.
     *
     * @param properties Configuration properties for the memory pool.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory pool is not supported for the
     * current CUDA version.
     */
    PinnedMemoryPool(PinnedPoolProperties properties = {});

    /**
     * @brief Constructs a new pinned memory pool.
     *
     * @param numa_id The optional NUMA node ID to associate with this pool.
     * @param properties Configuration properties for the memory pool.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory pool is not supported for the
     * current CUDA version.
     */
    PinnedMemoryPool(int numa_id, PinnedPoolProperties properties = {});

    /**
     * @brief Destroys the pinned memory pool.
     *
     * Releases all memory associated with this pool.
     */
    ~PinnedMemoryPool();

    /**
     * @brief Gets the properties used to configure this pool.
     *
     * @return A const reference to the pool properties.
     */
    [[nodiscard]] constexpr PinnedPoolProperties const& properties() const noexcept {
        return properties_;
    }

  private:
    PinnedPoolProperties properties_;  ///< Configuration properties for this pool.

    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryPoolImpl;
    std::unique_ptr<PinnedMemoryPoolImpl> impl_;
};

/// @brief The default alignment for pinned memory allocations.
constexpr size_t default_pinned_memory_alignment =
    cuda::mr::default_cuda_malloc_alignment;

/**
 * @brief A memory resource that allocates/deallocates pinned host memory from a pinned
 * host memory pool. This internally uses
 * `cuda::experimental::pinned_memory_resource`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__resource.html
 *
 * This class provides an interface for allocating and deallocating pinned
 * (page-locked) host memory asynchronously using CUDA streams.
 */
class PinnedMemoryResource {
  public:
    /**
     * @brief Friend function to get the host_accessible property.
     */
    friend constexpr void get_property(
        const PinnedMemoryResource&, cuda::mr::host_accessible
    ) noexcept {}

    /**
     * @brief Constructs a new pinned memory resource.
     *
     * @param pool The pinned memory pool to use for allocations.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory resource is not supported for
     * the current CUDA version.
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
     * @param stream The CUDA stream to use for the allocation operation.
     * @param bytes The number of bytes to allocate.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate(rmm::cuda_stream_view stream, size_t bytes);

    /**
     * @brief Allocates pinned memory asynchronously with alignment.
     *
     * @param stream The CUDA stream to use for the allocation operation.
     * @param bytes The number of bytes to allocate.
     * @param alignment The alignment requirement for the allocation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate(rmm::cuda_stream_view stream, size_t bytes, size_t alignment);

    /**
     * @brief Allocates pinned memory synchronously.
     *
     * @param bytes The number of bytes to allocate.
     * @param alignment The alignment requirement for the allocation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate_sync(size_t bytes, size_t alignment = default_pinned_memory_alignment);

    /**
     * @brief Deallocates pinned memory asynchronously.
     *
     * @param stream The CUDA stream to use for the deallocation operation.
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     */
    void deallocate(rmm::cuda_stream_view stream, void* ptr, size_t bytes) noexcept;

    /**
     * @brief Deallocates pinned memory asynchronously with alignment.
     *
     * @param stream The CUDA stream to use for the deallocation operation.
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     * @param alignment The alignment that was used for allocation.
     */
    void deallocate(
        rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
    ) noexcept;

    /**
     * @brief Deallocates pinned memory synchronously with alignment.
     *
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     * @param alignment The alignment that was used for allocation.
     */
    void deallocate_sync(
        void* ptr, size_t bytes, size_t alignment = default_pinned_memory_alignment
    );

    /**
     * @brief equality operator
     *
     * @param other The other pinned memory resource to compare with.
     * @return True if the two pinned memory resources are equal, false otherwise.
     */
    bool operator==(const PinnedMemoryResource& other) const noexcept {
        return impl_ == other.impl_;
    }

  private:
    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryResourceImpl;
    std::unique_ptr<PinnedMemoryResourceImpl> impl_;
};

static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::host_accessible>);

/**
 * @brief A buffer that manages stream-ordered pinned host memory. Only available for CUDA
 * versions >= 12.6.
 *
 * @note The buffer is allocated asynchronously on a given stream. Even though `data()`
 * ptr is immediately available, the buffer may not be ready to use in stream-unaware
 * operations until the stream is synchronized. Use `stream()` to get the stream
 * view and synchronize it as needed.
 *
 * \sa
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g871003f518e27ec92f7b331307fa32d4
 *
 * @code{.cpp}
 * rapidsmpf::PinnedHostBuffer buffer1(1024, stream, mr);
 * rapidsmpf::PinnedHostBuffer buffer2(1024, stream, mr);
 *
 * // data() is immediately available for cuda*Async operations. Eg.
 * cudaMemcpyAsync(buffer2.data(), buffer1.data(), 1024, cudaMemcpyDefault, stream);
 *
 * // for stream-unaware operations, the stream must be synchronized.
 * std::vector<uint8_t> data(1024);
 *
 * // synchronize buffer1
 * buffer1.synchronize();
 *
 * // now buffer1 is ready to use in stream-unaware operations.
 * std::memcpy(data.data(), buffer1.data(), 1024);
 * std::cout << buffer1.data()[0] << ", " << data[0] << std::endl;
 * @endcode
 */
class PinnedHostBuffer {
  public:
    PinnedHostBuffer() = default;

    /**
     * @brief Constructs an empty pinned host buffer.
     *
     * @param size The size of the buffer in bytes.
     * @param stream The CUDA stream to use for memory operations.
     * @param mr Memory resource to use for allocation and deallocation.
     *
     * @throws std::invalid_argument If @p mr is nullptr.
     */
    PinnedHostBuffer(
        size_t size,
        rmm::cuda_stream_view stream,
        std::shared_ptr<PinnedMemoryResource> mr
    );

    /**
     * @brief Constructs a pinned host buffer and asynchronously copies data into it.
     *
     * @param src_data Pointer to the source data to copy. If the data ptr is ordered on a
     * stream other than @p stream, it must be synchronized with @p stream before calling
     * this constructor. Otherwise, it will result in undefined behavior.
     * @param size The size of the data to copy in bytes.
     * @param stream The CUDA stream to use for memory operations.
     * @param mr Memory resource to use for allocation and deallocation.
     *
     * @throws std::invalid_argument If @p src_data is nullptr or @p data_ is nullptr (ie.
     * allocation failed).
     * @throws rapidsmpf::cuda_error If @p cudaMemcpyAsync fails.
     */
    PinnedHostBuffer(
        void const* src_data,
        size_t size,
        rmm::cuda_stream_view stream,
        std::shared_ptr<PinnedMemoryResource> mr
    );

    /**
     * @brief Constructs a pinned host buffer by copying data asynchronously from another
     * buffer on the same stream as @p other.
     *
     * @tparam OtherBufferT The type of the other buffer to copy from. Eg.
     * `rmm::device_buffer` or `PinnedHostBuffer`.
     *
     * @param other The other pinned host buffer to copy from.
     * @param mr Memory resource to use for allocation and deallocation.
     */
    template <typename OtherBufferT>
    PinnedHostBuffer(OtherBufferT const& other, std::shared_ptr<PinnedMemoryResource> mr)
        : PinnedHostBuffer(other.data(), other.size(), other.stream(), std::move(mr)) {}

    /**
     * @brief Asynchronously destroys the pinned host buffer.
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
     * @return The CUDA stream view.
     */
    [[nodiscard]] constexpr rmm::cuda_stream_view stream() const noexcept {
        return stream_;
    }

    /**
     * @brief Sets the CUDA stream for this buffer.
     *
     * @note This operation does not synchronize current `stream_` before setting the
     * @p stream.
     *
     * @param stream The new CUDA stream to use for memory operations.
     */
    constexpr void set_stream(rmm::cuda_stream_view stream) noexcept {
        stream_ = stream;
    }

    /**
     * @brief Synchronizes the buffer with the underlying stream.
     *
     * @throws rapidsmpf::cuda_error if the stream synchronization fails.
     */
    void synchronize();

  private:
    std::byte* data_{nullptr};  ///< Pointer to the allocated buffer data.
    size_t size_;  ///< Size of the buffer in bytes.
    rmm::cuda_stream_view stream_;  ///< CUDA stream used for memory operations.
    std::shared_ptr<PinnedMemoryResource>
        mr_;  ///< memory resource used for allocation/deallocation.
};

}  // namespace rapidsmpf
