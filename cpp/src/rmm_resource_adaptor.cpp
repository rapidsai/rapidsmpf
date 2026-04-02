/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {

// ---------------------------------------------------------------------------
// detail::RmmResourceAdaptorImpl
// ---------------------------------------------------------------------------
namespace detail {

RmmResourceAdaptorImpl::RmmResourceAdaptorImpl(
    rmm::device_async_resource_ref primary_mr,
    std::optional<rmm::device_async_resource_ref> fallback_mr
)
    : primary_mr_{std::move(primary_mr)}, fallback_mr_{std::move(fallback_mr)} {}

bool RmmResourceAdaptorImpl::operator==(
    RmmResourceAdaptorImpl const& other
) const noexcept {
    if (this == &other) {
        return true;
    }
    // Manual comparison of optionals to avoid recursive constraint satisfaction in
    // CCCL 3.2. std::optional::operator== triggers infinite concept checking when the
    // wrapped type (rmm::device_async_resource_ref) inherits from CCCL's concept-based
    // resource_ref.
    // TODO: Revert this after the RMM resource ref types are replaced with
    // plain cuda::mr ref types. This depends on
    // https://github.com/rapidsai/rmm/issues/2011.
    auto this_fallback = get_fallback_resource();
    auto other_fallback = other.get_fallback_resource();
    bool fallbacks_equal =
        (this_fallback.has_value() == other_fallback.has_value())
        && (!this_fallback.has_value() || (*this_fallback == *other_fallback));
    return get_upstream_resource() == other.get_upstream_resource() && fallbacks_equal;
}

rmm::device_async_resource_ref
RmmResourceAdaptorImpl::get_upstream_resource() const noexcept {
    return primary_mr_;
}

std::optional<rmm::device_async_resource_ref>
RmmResourceAdaptorImpl::get_fallback_resource() const noexcept {
    return fallback_mr_;
}

ScopedMemoryRecord RmmResourceAdaptorImpl::get_main_record() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_record_;
}

std::int64_t RmmResourceAdaptorImpl::current_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_record_.current();
}

void RmmResourceAdaptorImpl::begin_scoped_memory_record() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Push an empty scope on the stack.
    record_stacks_[std::this_thread::get_id()].emplace();
}

ScopedMemoryRecord RmmResourceAdaptorImpl::end_scoped_memory_record() {
    std::lock_guard lock(mutex_);
    auto& stack = record_stacks_.at(std::this_thread::get_id());
    RAPIDSMPF_EXPECTS(
        !stack.empty(),
        "calling end_scoped_memory_record() on an empty stack.",
        std::out_of_range
    );
    auto ret = stack.top();
    stack.pop();
    if (!stack.empty()) {
        // Add this ending scope to the new topmost scope.
        stack.top().add_subscope(ret);
    }
    return ret;
}

void* RmmResourceAdaptorImpl::allocate(
    cuda::stream_ref stream, std::size_t nbytes, std::size_t /*alignment*/
) {
    constexpr auto PRIMARY = ScopedMemoryRecord::AllocType::PRIMARY;
    constexpr auto FALLBACK = ScopedMemoryRecord::AllocType::FALLBACK;

    void* ret{};
    auto alloc_type = PRIMARY;
    try {
        ret = primary_mr_.allocate(stream, nbytes);
    } catch (rmm::out_of_memory const& e) {
        if (fallback_mr_.has_value()) {
            alloc_type = FALLBACK;
            ret = fallback_mr_->allocate(stream, nbytes);
            std::lock_guard<std::mutex> lock(mutex_);
            fallback_allocations_.insert(ret);
        } else {
            throw;
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);

    // Always record the allocation on the main record.
    main_record_.record_allocation(alloc_type, safe_cast<std::int64_t>(nbytes));

    // But only record the allocation on the thread stack, if `record_stacks_`
    // isn't empty i.e. someone has called `begin_scoped_memory_record`.
    if (!record_stacks_.empty()) {
        auto const thread_id = std::this_thread::get_id();
        auto& record = record_stacks_[thread_id];
        if (!record.empty()) {
            record.top().record_allocation(alloc_type, safe_cast<std::int64_t>(nbytes));
            RAPIDSMPF_EXPECTS(
                allocating_threads_.insert({ret, thread_id}).second,
                "duplicate memory pointer"
            );
        }
    }
    return ret;
}

void RmmResourceAdaptorImpl::deallocate(
    cuda::stream_ref stream, void* ptr, std::size_t nbytes, std::size_t /*alignment*/
) noexcept {
    constexpr auto PRIMARY = ScopedMemoryRecord::AllocType::PRIMARY;
    constexpr auto FALLBACK = ScopedMemoryRecord::AllocType::FALLBACK;

    ScopedMemoryRecord::AllocType alloc_type;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        alloc_type = (fallback_allocations_.erase(ptr) == 0) ? PRIMARY : FALLBACK;

        // Always record the deallocation on the main record.
        main_record_.record_deallocation(alloc_type, safe_cast<std::int64_t>(nbytes));
        // But only record it on the thread stack if it exist.
        if (!allocating_threads_.empty()) {
            auto const node = allocating_threads_.extract(ptr);
            if (node) {
                auto thread_id = node.mapped();  // `ptr` was allocated by `thread_id`.
                auto& record = record_stacks_[thread_id];
                if (!record.empty()) {
                    record.top().record_deallocation(
                        alloc_type, safe_cast<std::int64_t>(nbytes)
                    );
                }
            }
        }
    }
    if (alloc_type == PRIMARY) {
        primary_mr_.deallocate(stream, ptr, nbytes);
    } else {
        fallback_mr_->deallocate(stream, ptr, nbytes);
    }
}

void* RmmResourceAdaptorImpl::allocate_sync(std::size_t bytes, std::size_t alignment) {
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
    return ptr;
}

void RmmResourceAdaptorImpl::deallocate_sync(
    void* ptr, std::size_t bytes, std::size_t alignment
) noexcept {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// RmmResourceAdaptor (thin shell delegating to shared_resource<Impl>)
// ---------------------------------------------------------------------------

RmmResourceAdaptor::RmmResourceAdaptor(
    rmm::device_async_resource_ref primary_mr,
    std::optional<rmm::device_async_resource_ref> fallback_mr
)
    : shared_base(
          cuda::mr::make_shared_resource<detail::RmmResourceAdaptorImpl>(
              primary_mr, fallback_mr
          )
      ) {}

rmm::device_async_resource_ref
RmmResourceAdaptor::get_upstream_resource() const noexcept {
    return get().get_upstream_resource();
}

std::optional<rmm::device_async_resource_ref>
RmmResourceAdaptor::get_fallback_resource() const noexcept {
    return get().get_fallback_resource();
}

ScopedMemoryRecord RmmResourceAdaptor::get_main_record() const {
    return get().get_main_record();
}

std::int64_t RmmResourceAdaptor::current_allocated() const noexcept {
    return get().current_allocated();
}

void RmmResourceAdaptor::begin_scoped_memory_record() {
    get().begin_scoped_memory_record();
}

ScopedMemoryRecord RmmResourceAdaptor::end_scoped_memory_record() {
    return get().end_scoped_memory_record();
}

}  // namespace rapidsmpf
