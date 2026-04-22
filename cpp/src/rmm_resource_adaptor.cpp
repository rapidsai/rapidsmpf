/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/rmm_resource_adaptor.hpp>

namespace rapidsmpf {

RmmResourceAdaptor::RmmResourceAdaptor(
    cuda::mr::any_resource<cuda::mr::device_accessible> primary_mr,
    std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> fallback_mr
)
    : shared_base(
          cuda::mr::make_shared_resource<
              detail::RmmResourceAdaptorImpl<any_device_resource>>(
              std::move(primary_mr), std::move(fallback_mr)
          )
      ) {}

rmm::device_async_resource_ref
RmmResourceAdaptor::get_upstream_resource() const noexcept {
    return rmm::device_async_resource_ref{
        const_cast<any_device_resource&>(get().get_upstream_resource())
    };
}

std::optional<rmm::device_async_resource_ref>
RmmResourceAdaptor::get_fallback_resource() const noexcept {
    auto const& fallback = get().get_fallback_resource();
    if (fallback.has_value()) {
        return rmm::device_async_resource_ref{
            const_cast<any_device_resource&>(*fallback)
        };
    }
    return std::nullopt;
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
