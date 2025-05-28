/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {


void ScopedMemoryRecord::record_allocation(AllocType alloc_type, std::uint64_t nbytes) {
    auto at = static_cast<std::size_t>(alloc_type);
    ++num_allocs_[at];
    current_[at] += nbytes;
    total_[at] += nbytes;
    peak_[at] = std::max(peak_[at], current_[at]);
    highest_peak_ = std::max(highest_peak_, current_[at]);
}

void ScopedMemoryRecord::record_deallocation(AllocType alloc_type, std::uint64_t nbytes) {
    auto at = static_cast<std::size_t>(alloc_type);
    current_[at] -= nbytes;
}

std::uint64_t RmmResourceAdaptor::current_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return record_.current(ScopedMemoryRecord::AllocType::Primary)
           + record_.current(ScopedMemoryRecord::AllocType::Fallback);
}

void* RmmResourceAdaptor::do_allocate(std::size_t nbytes, rmm::cuda_stream_view stream) {
    void* ret{};
    try {
        ret = primary_mr_.allocate_async(nbytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        record_.record_allocation(ScopedMemoryRecord::AllocType::Primary, nbytes);
    } catch (rmm::out_of_memory const& e) {
        if (fallback_mr_.has_value()) {
            ret = fallback_mr_->allocate_async(nbytes, stream);
            std::lock_guard<std::mutex> lock(mutex_);
            fallback_allocations_.insert(ret);
            record_.record_allocation(ScopedMemoryRecord::AllocType::Fallback, nbytes);
        } else {
            throw;
        }
    }
    return ret;
}

namespace {
// If it exist, erase fallback allocation and return true else return false.
bool erase_fallback_allocation(
    std::mutex& mutex,
    std::optional<rmm::device_async_resource_ref> const& fallback_mr,
    std::unordered_set<void*>& fallback_allocations,
    void* ptr
) {
    if (fallback_mr.has_value()) {
        std::lock_guard<std::mutex> lock(mutex);
        return fallback_allocations.erase(ptr) == 1;
    }
    return false;
}
}  // namespace

void RmmResourceAdaptor::do_deallocate(
    void* ptr, std::size_t nbytes, rmm::cuda_stream_view stream
) {
    if (erase_fallback_allocation(mutex_, fallback_mr_, fallback_allocations_, ptr)) {
        fallback_mr_->deallocate_async(ptr, nbytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        record_.record_deallocation(ScopedMemoryRecord::AllocType::Fallback, nbytes);
    } else {
        primary_mr_.deallocate_async(ptr, nbytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        record_.record_deallocation(ScopedMemoryRecord::AllocType::Primary, nbytes);
    }
}

bool RmmResourceAdaptor::do_is_equal(rmm::mr::device_memory_resource const& other
) const noexcept {
    if (this == &other) {
        return true;
    }
    auto cast = dynamic_cast<RmmResourceAdaptor const*>(&other);
    if (cast == nullptr) {
        return false;
    }
    return get_upstream_resource() == cast->get_upstream_resource()
           && get_fallback_resource() == cast->get_fallback_resource();
}
}  // namespace rapidsmpf
