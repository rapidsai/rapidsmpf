/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf::detail {

/**
 * @brief Implementation class for RmmResourceAdaptor.
 *
 * Holds all mutable state for memory tracking and fallback allocation.
 * This class satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `RmmResourceAdaptor` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class RmmResourceAdaptorImpl {
  public:
    /**
     * @brief Construct with primary and optional fallback memory resource.
     *
     * @param primary_mr The primary memory resource.
     * @param fallback_mr Optional fallback memory resource.
     */
    RmmResourceAdaptorImpl(
        cuda::mr::any_resource<cuda::mr::device_accessible> primary_mr,
        std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> fallback_mr
    );

    ~RmmResourceAdaptorImpl() = default;

    RmmResourceAdaptorImpl(RmmResourceAdaptorImpl const&) = delete;
    RmmResourceAdaptorImpl(RmmResourceAdaptorImpl&&) = delete;
    RmmResourceAdaptorImpl& operator=(RmmResourceAdaptorImpl const&) = delete;
    RmmResourceAdaptorImpl& operator=(RmmResourceAdaptorImpl&&) = delete;

    /**
     * @brief Equality comparison (identity-based).
     *
     * @param other The other impl to compare.
     * @return True if this and other are the same object.
     */
    [[nodiscard]] bool operator==(RmmResourceAdaptorImpl const& other) const noexcept;

    /// @copydoc RmmResourceAdaptor::get_upstream_resource
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

    /// @copydoc RmmResourceAdaptor::get_fallback_resource
    [[nodiscard]] std::optional<rmm::device_async_resource_ref>
    get_fallback_resource() const noexcept;

    /// @copydoc RmmResourceAdaptor::get_main_record
    [[nodiscard]] ScopedMemoryRecord get_main_record() const;

    /// @copydoc RmmResourceAdaptor::current_allocated
    [[nodiscard]] std::int64_t current_allocated() const noexcept;

    /// @copydoc RmmResourceAdaptor::begin_scoped_memory_record
    void begin_scoped_memory_record();

    /// @copydoc RmmResourceAdaptor::end_scoped_memory_record
    ScopedMemoryRecord end_scoped_memory_record();

    /**
     * @brief Allocate memory asynchronously on the given stream.
     *
     * @param stream The CUDA stream for the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    );

    /**
     * @brief Deallocate memory asynchronously on the given stream.
     *
     * @param stream The CUDA stream for the deallocation.
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes to deallocate.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept;

    /**
     * @brief Allocate memory synchronously.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     */
    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    );

    /**
     * @brief Deallocate memory synchronously.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes to deallocate.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate_sync(
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept;

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        RmmResourceAdaptorImpl const&, cuda::mr::device_accessible
    ) noexcept {}

  private:
    mutable std::mutex mutex_;
    cuda::mr::any_resource<cuda::mr::device_accessible> primary_mr_;
    std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> fallback_mr_;
    std::unordered_set<void*> fallback_allocations_;

    ScopedMemoryRecord main_record_;
    std::unordered_map<std::thread::id, std::stack<ScopedMemoryRecord>> record_stacks_;
    std::unordered_map<void*, std::thread::id> allocating_threads_;
};

}  // namespace rapidsmpf::detail
