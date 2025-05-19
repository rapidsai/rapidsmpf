/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_set>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace rapidsmpf {


/**
 * @brief A device memory resource that uses an alternate upstream resource when the
 * primary upstream resource throws a specified exception type.
 *
 * An instance of this resource must be constructed with two upstream resources to satisfy
 * allocation requests.
 *
 * @tparam ExceptionType The type of exception that this adaptor should respond to.
 */
template <typename ExceptionType = rmm::out_of_memory>
class RmmFallbackResource final : public rmm::mr::device_memory_resource {
  public:
    using exception_type =
        ExceptionType;  ///< The type of exception this object catches/throws

    /**
     * @brief Construct a new `RmmFallbackResource` that uses `primary_upstream`
     * to satisfy allocation requests and if that fails with `ExceptionType`, uses
     * `alternate_upstream`.
     *
     * @param primary_upstream The primary resource used for allocating/deallocating
     * device memory
     * @param alternate_upstream The alternate resource used for allocating/deallocating
     * device memory memory
     */
    RmmFallbackResource(
        rmm::device_async_resource_ref primary_upstream,
        rmm::device_async_resource_ref alternate_upstream
    )
        : primary_upstream_{primary_upstream}, alternate_upstream_{alternate_upstream} {}

    RmmFallbackResource() = delete;
    ~RmmFallbackResource() override = default;

    /**
     * @brief Move constructor for RmmFallbackResource.
     *
     * @param other The RmmFallbackResource instance to move from.
     * @return Reference to the moved instance.
     */
    RmmFallbackResource& operator=(RmmFallbackResource&& other) noexcept = default;

    /**
     * @brief Get a reference to the primary upstream resource.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept {
        return primary_upstream_;
    }

    RmmFallbackResource(RmmFallbackResource const&) = delete;
    RmmFallbackResource& operator=(RmmFallbackResource const&) = delete;

    /**
     * @brief Get a reference to the alternative upstream resource.
     *
     * This resource is used when primary upstream resource throws `exception_type`.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_alternate_upstream_resource(
    ) const noexcept {
        return alternate_upstream_;
    }

  private:
    /**
     * @brief Allocates memory of size at least `bytes` using the upstream
     * resource.
     *
     * @throws any exceptions thrown from the upstream resources, only `exception_type`
     * thrown by the primary upstream is caught.
     *
     * @param bytes The size, in bytes, of the allocation
     * @param stream Stream on which to perform the allocation
     * @return void* Pointer to the newly allocated memory
     */
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        void* ret{};
        try {
            ret = primary_upstream_.allocate_async(bytes, stream);
        } catch (exception_type const& e) {
            ret = alternate_upstream_.allocate_async(bytes, stream);
            std::lock_guard<std::mutex> lock(mtx_);
            alternate_allocations_.insert(ret);
        }
        return ret;
    }

    /**
     * @brief Free allocation of size `bytes` pointed to by `ptr`
     *
     * @param ptr Pointer to be deallocated
     * @param bytes Size of the allocation
     * @param stream Stream on which to perform the deallocation
     */
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
        override {
        std::size_t count{0};
        {
            std::lock_guard<std::mutex> lock(mtx_);
            count = alternate_allocations_.erase(ptr);
        }
        if (count > 0) {
            alternate_upstream_.deallocate_async(ptr, bytes, stream);
        } else {
            primary_upstream_.deallocate_async(ptr, bytes, stream);
        }
    }

    /**
     * @brief Compare the resource to another.
     *
     * @param other The other resource to compare to
     * @return true If the two resources are equivalent
     * @return false If the two resources are not equal
     */
    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other
    ) const noexcept override {
        if (this == &other) {
            return true;
        }
        auto cast = dynamic_cast<RmmFallbackResource const*>(&other);
        if (cast == nullptr) {
            return false;
        }
        return get_upstream_resource() == cast->get_upstream_resource()
               && get_alternate_upstream_resource()
                      == cast->get_alternate_upstream_resource();
    }

    rmm::device_async_resource_ref primary_upstream_;
    rmm::device_async_resource_ref alternate_upstream_;
    std::unordered_set<void*> alternate_allocations_;
    mutable std::mutex mtx_;
};


}  // namespace rapidsmpf
