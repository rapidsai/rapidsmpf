/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {

ScopedMemoryRecord RmmResourceAdaptor::get_main_record() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_record_;
}

std::int64_t RmmResourceAdaptor::current_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_record_.current();
}

void RmmResourceAdaptor::begin_scoped_memory_record() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Push an empty scope on the stack.
    record_stacks_[std::this_thread::get_id()].emplace();
}

ScopedMemoryRecord RmmResourceAdaptor::end_scoped_memory_record() {
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

void* RmmResourceAdaptor::do_allocate(std::size_t nbytes, rmm::cuda_stream_view stream) {
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
    main_record_.record_allocation(alloc_type, static_cast<std::int64_t>(nbytes));

    // But only record the allocation on the thread stack, if `record_stacks_`
    // isn't empty i.e. someone has called `begin_scoped_memory_record`.
    if (!record_stacks_.empty()) {
        auto const thread_id = std::this_thread::get_id();
        auto& record = record_stacks_[thread_id];
        if (!record.empty()) {
            record.top().record_allocation(alloc_type, static_cast<std::int64_t>(nbytes));
            RAPIDSMPF_EXPECTS(
                allocating_threads_.insert({ret, thread_id}).second,
                "duplicate memory pointer"
            );
        }
    }
    return ret;
}

void RmmResourceAdaptor::do_deallocate(
    void* ptr, std::size_t nbytes, rmm::cuda_stream_view stream
) noexcept {
    constexpr auto PRIMARY = ScopedMemoryRecord::AllocType::PRIMARY;
    constexpr auto FALLBACK = ScopedMemoryRecord::AllocType::FALLBACK;

    ScopedMemoryRecord::AllocType alloc_type;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        alloc_type = (fallback_allocations_.erase(ptr) == 0) ? PRIMARY : FALLBACK;

        // Always record the deallocation on the main record.
        main_record_.record_deallocation(alloc_type, static_cast<std::int64_t>(nbytes));
        // But only record it on the thread stack if it exist.
        if (!allocating_threads_.empty()) {
            auto const node = allocating_threads_.extract(ptr);
            if (node) {
                auto thread_id = node.mapped();  // `ptr` was allocated by `thread_id`.
                auto& record = record_stacks_[thread_id];
                if (!record.empty()) {
                    record.top().record_deallocation(
                        alloc_type, static_cast<std::int64_t>(nbytes)
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

bool RmmResourceAdaptor::do_is_equal(
    rmm::mr::device_memory_resource const& other
) const noexcept {
    if (this == &other) {
        return true;
    }
    auto cast = dynamic_cast<RmmResourceAdaptor const*>(&other);
    if (cast == nullptr) {
        return false;
    }
    // Manual comparison of optionals to avoid recursive constraint satisfaction in
    // CCCL 3.2. std::optional::operator== triggers infinite concept checking when the
    // wrapped type (rmm::device_async_resource_ref) inherits from CCCL's concept-based
    // resource_ref.
    // TODO: Revert this after the RMM resource ref types are replaced with
    // plain cuda::mr ref types. This depends on
    // https://github.com/rapidsai/rmm/issues/2011.
    auto this_fallback = get_fallback_resource();
    auto other_fallback = cast->get_fallback_resource();
    bool fallbacks_equal =
        (this_fallback.has_value() == other_fallback.has_value())
        && (!this_fallback.has_value() || (*this_fallback == *other_fallback));
    return get_upstream_resource() == cast->get_upstream_resource() && fallbacks_equal;
}

}  // namespace rapidsmpf
