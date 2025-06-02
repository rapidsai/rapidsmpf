/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>

#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {

namespace {
/**
 * @brief Retrieves a value from a statistics array or accumulates the total.
 *
 * @param arr        The array containing statistics for each allocator type.
 * @param alloc_type The type of allocator to retrieve data for. If `AllocType::ALL`,
 *                   the function returns the sum across all entries in the array.
 * @return The requested statistic value or the accumulated total.
 */
std::uint64_t get_or_accumulate(
    ScopedMemoryRecord::AllocTypeArray const& arr,
    ScopedMemoryRecord::AllocType alloc_type
) noexcept {
    if (alloc_type == ScopedMemoryRecord::AllocType::ALL) {
        return std::accumulate(arr.begin(), arr.end(), std::uint64_t{0});
    }
    return arr[static_cast<std::size_t>(alloc_type)];
}
}  // namespace

std::uint64_t ScopedMemoryRecord::num_total_allocs(AllocType alloc_type) const noexcept {
    return get_or_accumulate(num_total_allocs_, alloc_type);
}

std::uint64_t ScopedMemoryRecord::num_current_allocs(AllocType alloc_type
) const noexcept {
    return get_or_accumulate(num_current_allocs_, alloc_type);
}

std::uint64_t ScopedMemoryRecord::current(AllocType alloc_type) const noexcept {
    return get_or_accumulate(current_, alloc_type);
}

std::uint64_t ScopedMemoryRecord::total(AllocType alloc_type) const noexcept {
    return get_or_accumulate(total_, alloc_type);
}

std::uint64_t ScopedMemoryRecord::peak(AllocType alloc_type) const noexcept {
    if (alloc_type == AllocType::ALL) {
        return highest_peak_;
    }
    return peak_[static_cast<std::size_t>(alloc_type)];
}

void ScopedMemoryRecord::record_allocation(AllocType alloc_type, std::uint64_t nbytes) {
    RAPIDSMPF_EXPECTS(
        alloc_type != AllocType::ALL,
        "AllocType::ALL may not be used to record allocation"
    );
    auto at = static_cast<std::size_t>(alloc_type);
    ++num_total_allocs_[at];
    ++num_current_allocs_[at];
    current_[at] += nbytes;
    total_[at] += nbytes;
    peak_[at] = std::max(peak_[at], current_[at]);
    highest_peak_ = std::max(highest_peak_, current());
}

void ScopedMemoryRecord::record_deallocation(AllocType alloc_type, std::uint64_t nbytes) {
    RAPIDSMPF_EXPECTS(
        alloc_type != AllocType::ALL,
        "AllocType::ALL may not be used to record deallocation"
    );
    auto at = static_cast<std::size_t>(alloc_type);
    current_[at] -= nbytes;
    --num_current_allocs_[at];
}

std::uint64_t RmmResourceAdaptor::current_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return record_.current();
}

void* RmmResourceAdaptor::do_allocate(std::size_t nbytes, rmm::cuda_stream_view stream) {
    void* ret{};
    try {
        ret = primary_mr_.allocate_async(nbytes, stream);
        std::lock_guard<std::mutex> lock(mutex_);
        record_.record_allocation(ScopedMemoryRecord::AllocType::PRIMARY, nbytes);
    } catch (rmm::out_of_memory const& e) {
        if (fallback_mr_.has_value()) {
            ret = fallback_mr_->allocate_async(nbytes, stream);
            std::lock_guard<std::mutex> lock(mutex_);
            fallback_allocations_.insert(ret);
            record_.record_allocation(ScopedMemoryRecord::AllocType::FALLBACK, nbytes);
        } else {
            throw;
        }
    }
    return ret;
}

void RmmResourceAdaptor::do_deallocate(
    void* ptr, std::size_t nbytes, rmm::cuda_stream_view stream
) {
    std::unique_lock lock(mutex_);
    if (fallback_allocations_.erase(ptr) == 1)
    {  // ptr was allocated from fallback mr and fallback mr is available
        record_.record_deallocation(ScopedMemoryRecord::AllocType::FALLBACK, nbytes);
        lock.unlock();
        fallback_mr_->deallocate_async(ptr, nbytes, stream);
    } else {
        record_.record_deallocation(ScopedMemoryRecord::AllocType::PRIMARY, nbytes);
        lock.unlock();
        primary_mr_.deallocate_async(ptr, nbytes, stream);
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
