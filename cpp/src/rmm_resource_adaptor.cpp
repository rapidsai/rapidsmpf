/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {


std::uint64_t RmmResourceAdaptor::current_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return primary_record_.current + fallback_record_.current;
}

void* RmmResourceAdaptor::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    void* ret{};
    try {
        ret = primary_mr_.allocate_async(bytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        primary_record_.record_allocation(bytes);
    } catch (rmm::out_of_memory const& e) {
        if (fallback_mr_.has_value()) {
            ret = fallback_mr_->allocate_async(bytes, stream);
            std::lock_guard<std::mutex> lock(mutex_);
            fallback_allocations_.insert(ret);
            fallback_record_.record_allocation(bytes);
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
    void* ptr, std::size_t bytes, rmm::cuda_stream_view stream
) {
    if (erase_fallback_allocation(mutex_, fallback_mr_, fallback_allocations_, ptr)) {
        fallback_mr_->deallocate_async(ptr, bytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        fallback_record_.record_deallocation(bytes);
    } else {
        primary_mr_.deallocate_async(ptr, bytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        primary_record_.record_deallocation(bytes);
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
