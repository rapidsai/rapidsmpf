/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <rmm/aligned.hpp>

namespace rapidsmpf {

/**
 * @brief CCCL-compatible memory resource adaptor that makes copies keep an
 * external owner alive.
 *
 * Wraps another CCCL-compatible resource and forwards allocation,
 * deallocation, and property queries to it. The adaptor is useful when an
 * object owns runtime state needed by the wrapped resource, but exposes that
 * resource through a non-owning `cuda::mr::resource_ref`.
 *
 * This makes it safe for downstream code to promote that ref to an owning
 * `cuda::mr::any_resource`: the copied adaptor stored in the
 * `any_resource` keeps the back-referenced owner alive.
 *
 * Properties exposed by the wrapped resource through `get_property` ADL are
 * forwarded transparently via `cuda::forward_property`.
 *
 * @tparam Resource Wrapped CCCL-compatible resource type. Must satisfy the
 * `cuda::mr::resource` concept and be copyable.
 * @tparam BackRef Type of the back-referenced owner object.
 *
 * @par CCCL's lifetime semantic
 *
 * `cuda::mr::resource_ref<...>` is a non-owning type-erased view. It stores a
 * pointer to a concrete resource object together with a vtable. Copying a
 * `resource_ref` only copies that pointer and vtable. It does not copy the
 * resource and does not extend its lifetime. The caller must ensure that the
 * referenced resource object remains alive for every direct use of the ref.
 *
 * `cuda::mr::any_resource<...>` is an owning type-erased resource. Constructing
 * an `any_resource` from a `resource_ref` performs a deep copy of the
 * referenced concrete resource into storage owned by the `any_resource`. That
 * deep copy invokes the concrete resource's copy constructor.
 *
 * A deep copy of the resource is not necessarily enough to make the result
 * self-contained. If the copied resource depends on state owned outside the
 * resource object, for example through a back-pointer, reference, or shared
 * runtime state, then the copy preserves that dependency. Without additional
 * lifetime management, the original owner can be destroyed while the copied
 * resource inside the `any_resource` still depends on it.
 *
 * This adaptor makes that copy step acquire ownership of the back-referenced
 * owner. The original adaptor normally lives inside `*back_ref` and stores
 * only a `std::weak_ptr<BackRef>`, avoiding a reference cycle. When CCCL copies
 * the adaptor, such as during `any_resource(resource_ref)` construction, the
 * adaptor's copy constructor promotes the weak reference to a
 * `std::shared_ptr<BackRef>`. The copied adaptor therefore keeps the owner
 * alive for as long as the copied resource exists.
 *
 * If the owner has already been destroyed when the adaptor is copied, the copy
 * constructor throws `std::bad_weak_ptr` instead of producing a dangling
 * adaptor.
 *
 * This does not make direct use of a bare `resource_ref` safe after the owner
 * has been destroyed. A call such as `ref.allocate(...)` still dispatches
 * through the `resource_ref` into the original resource object, so the owner
 * must outlive all direct uses of the non-owning ref.
 */
template <typename Resource, typename BackRef>
class OwningResourceAdaptor
    : public cuda::forward_property<OwningResourceAdaptor<Resource, BackRef>, Resource> {
  public:
    /**
     * @brief Construct an adaptor that weakly references @p back_ref.
     *
     * The original adaptor instance typically lives inside `*back_ref` and stores only a
     * `std::weak_ptr<BackRef>`, avoiding a reference cycle. Copies of the adaptor promote
     * that weak reference to a `std::shared_ptr<BackRef>`, allowing copied resources to
     * keep the owner alive.
     *
     * @param resource Wrapped resource.
     * @param back_ref Weak reference to the back-referenced owner object.
     */
    OwningResourceAdaptor(Resource resource, std::weak_ptr<BackRef> back_ref) noexcept
        : resource_{std::move(resource)}, weak_{std::move(back_ref)} {}

    /**
     * @brief Copy constructor that promotes the back-reference to shared ownership.
     *
     * This constructor is the key lifetime hook used by `any_resource(resource_ref)`.
     * When CCCL deep-copies the wrapped resource into an `any_resource`, it invokes this
     * copy constructor. The copied adaptor acquires a `std::shared_ptr<BackRef>`,
     * ensuring the back-referenced owner stays alive for as long as the copied resource
     * exists.
     *
     * If @p other already holds a strong back-reference, it is reused. Otherwise the weak
     * reference is promoted via `std::shared_ptr` construction.
     *
     * @param other Adaptor to copy from.
     * @throws std::bad_weak_ptr if the weak back-reference has expired.
     */
    OwningResourceAdaptor(OwningResourceAdaptor const& other)
        : resource_{other.resource_},
          weak_{other.weak_},
          strong_{
              other.strong_ ? other.strong_
                            : std::shared_ptr<BackRef>{other.weak_}  // throws if expired
          } {}

    /**
     * @brief Copy assignment operator that promotes the back-reference to shared
     * ownership.
     *
     * Uses the same lifetime semantics as the copy constructor. After the assignment,
     * this adaptor holds a strong reference to the back-referenced owner object.
     *
     * @param other Adaptor to copy from.
     * @return Reference to this adaptor.
     * @throws std::bad_weak_ptr if the weak back-reference has expired.
     */
    OwningResourceAdaptor& operator=(OwningResourceAdaptor const& other) {
        if (this != &other) {
            resource_ = other.resource_;
            weak_ = other.weak_;
            strong_ = other.strong_
                          ? other.strong_
                          : std::shared_ptr<BackRef>{other.weak_};  // throws if expired
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * @param other Adaptor to move from.
     */
    OwningResourceAdaptor(OwningResourceAdaptor&& other) noexcept = default;

    /**
     * @brief Move assignment operator.
     *
     * @param other Adaptor to move from.
     * @return Reference to this adaptor.
     */
    OwningResourceAdaptor& operator=(OwningResourceAdaptor&& other) noexcept = default;

    ~OwningResourceAdaptor() = default;

    /**
     * @brief Accessor used by `cuda::forward_property`.
     *
     * Property queries received through `cuda::forward_property` are forwarded to the
     * wrapped resource returned by this accessor.
     *
     * @return Reference to the wrapped resource.
     */
    [[nodiscard]] Resource& get() noexcept {
        return resource_;
    }

    /** @copydoc get() */
    [[nodiscard]] Resource const& get() const noexcept {
        return resource_;
    }

    /**
     * @brief Allocate memory on a CUDA stream. Forwards to the wrapped resource.
     *
     * @param stream CUDA stream associated with the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Required allocation alignment.
     * @return Pointer to the allocated memory.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return resource_.allocate(stream, bytes, alignment);
    }

    /**
     * @brief Deallocate memory on a CUDA stream. Forwards to the wrapped resource.
     *
     * @param stream CUDA stream associated with the deallocation.
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes originally allocated.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        resource_.deallocate(stream, ptr, bytes, alignment);
    }

    /**
     * @brief Allocate memory synchronously. Forwards to the wrapped resource.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Required allocation alignment.
     * @return Pointer to the allocated memory.
     */
    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return resource_.allocate_sync(bytes, alignment);
    }

    /**
     * @brief Deallocate memory synchronously. Forwards to the wrapped resource.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes originally allocated.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate_sync(
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        resource_.deallocate_sync(ptr, bytes, alignment);
    }

    /**
     * @brief Compare wrapped resource identity.
     *
     * Two adaptors compare equal if their wrapped resources compare equal.
     *
     * @param other Adaptor to compare against.
     * @return `true` if both adaptors refer to the same underlying resource.
     */
    [[nodiscard]] bool operator==(OwningResourceAdaptor const& other) const noexcept {
        return resource_ == other.resource_;
    }

  private:
    Resource resource_;
    std::weak_ptr<BackRef> weak_;
    std::shared_ptr<BackRef> strong_;
};

}  // namespace rapidsmpf
