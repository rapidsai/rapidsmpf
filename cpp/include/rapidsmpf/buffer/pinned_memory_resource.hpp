/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>

/// @brief The minimum CUDA version required for PinnedMemoryResource.
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION 12600
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR \
    RAPIDSMPF_STRINGIFY(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)

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

    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
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
     * @param stream The CUDA stream to use for the allocation operation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate_async(size_t bytes, rmm::cuda_stream_view stream);

    /**
     * @brief Deallocates pinned memory asynchronously.
     *
     * @param ptr A pointer to the memory to deallocate.
     * @param stream The CUDA stream to use for the deallocation operation.
     */
    void deallocate_async(void* ptr, rmm::cuda_stream_view stream);

  private:
    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryResourceImpl;
    std::unique_ptr<PinnedMemoryResourceImpl> impl_;
};

/**
 * @brief A buffer that manages stream-ordered pinned host memory. Only available for CUDA
 * versions >= 12.6.
 *
 * @note The buffer is allocated asynchronously on a given stream. Even though `data()`
 * ptr is immediately available, the buffer may not be ready to use in stream-unaware
 * operations until the stream is synchronized. Use `stream()` to get the stream
 * view and synchronize it as needed. See for more details:
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
     * @throws std::invalid_argument If @p src_data is nullptr.
     * @throws std::invalid_argument If @p data_ is nullptr (ie. allocation failed).
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
     * pinned host buffer on the same stream.
     *
     * @param other The other pinned host buffer to copy from.
     * @param mr Memory resource to use for allocation and deallocation.
     */
    PinnedHostBuffer(
        PinnedHostBuffer const& other, std::shared_ptr<PinnedMemoryResource> mr
    )
        : PinnedHostBuffer(other.data(), other.size(), other.stream(), std::move(mr)) {}

    /**
     * @brief Constructs a pinned host buffer by copying data asynchronously from another
     * device buffer on the same stream.
     *
     * @param other The other device buffer to copy from.
     * @param mr Memory resource to use for allocation and deallocation.
     */
    PinnedHostBuffer(
        rmm::device_buffer const& other, std::shared_ptr<PinnedMemoryResource> mr
    )
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
    std::byte* data_ = nullptr;  ///< Pointer to the allocated buffer data.
    size_t size_;  ///< Size of the buffer in bytes.
    rmm::cuda_stream_view stream_;  ///< CUDA stream used for memory operations.
    std::shared_ptr<PinnedMemoryResource>
        mr_;  ///< memory resource used for allocation/deallocation.
};

}  // namespace rapidsmpf
