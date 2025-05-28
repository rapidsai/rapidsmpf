/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {

void* RmmResourceAdaptor::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    void* ret{};
    try {
        ret = primary_mr_.allocate_async(bytes, stream);
    } catch (rmm::out_of_memory const& e) {
        if (fallback_mr_.has_value()) {
            ret = fallback_mr_->allocate_async(bytes, stream);
            std::lock_guard<std::mutex> lock(mutex_);
            fallback_allocations_.insert(ret);
        } else {
            throw;
        }
    }
    return ret;
}

void RmmResourceAdaptor::do_deallocate(
    void* ptr, std::size_t bytes, rmm::cuda_stream_view stream
) {
    if (fallback_mr_.has_value()) {
        std::size_t count{0};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count = fallback_allocations_.erase(ptr);
        }
        if (count > 0) {
            fallback_mr_->deallocate_async(ptr, bytes, stream);
            return;
        }
    }
    primary_mr_.deallocate_async(ptr, bytes, stream);
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
