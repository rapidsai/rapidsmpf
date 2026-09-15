/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <concepts>
#include <memory>
#include <ranges>

#include <cuda/stream>

#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

namespace detail {

/**
 * @brief Internal concept for identifying ranges of a given type.
 *
 * @tparam R Range type.
 * @tparam T Expected content type.
 */
template <typename R, typename T>
concept input_range_of = std::ranges::input_range<R>
                         && std::convertible_to<std::ranges::range_reference_t<R>, T>;

}  // namespace detail

/**
 * @brief Make downstream CUDA streams wait on upstream CUDA streams.
 *
 * This call is asynchronous with respect to the host thread; no host-side
 * blocking occurs.
 *
 * @tparam Range1 Iterable whose elements are cuda::stream_ref.
 * @tparam Range2 Iterable whose elements are cuda::stream_ref.
 *
 * @param downstreams Streams that must not run ahead.
 * @param upstreams Streams whose already-enqueued work must complete first.
 * @param event Optional CUDA event used for synchronization. A unique event per
 * call is not required; the same event may be reused. If `nullptr`, a temporary
 * event is created internally. The reason to provide an event is to avoid the
 * small overhead of constructing a temporary one.
 *
 * @note If all upstream and downstream streams are identical, this function is a no-op.
 */
template <
    detail::input_range_of<cuda::stream_ref> Range1,
    detail::input_range_of<cuda::stream_ref> Range2>
void cuda_stream_join(
    Range1 const& downstreams, Range2 const& upstreams, CudaEvent* event = nullptr
) {
    // Quick exit if all streams are identical.
    if ([&] {
            for (cuda::stream_ref const& upstream : upstreams) {
                for (cuda::stream_ref const& downstream : downstreams) {
                    if (upstream.get() != downstream.get()) {
                        return false;
                    }
                }
            }
            return true;
        }())
    {
        return;
    }

    // Create a temporary CUDA event if none was provided. Note, once the event
    // has been used to record synchronization between streams, it can be safely
    // destroyed without affecting the synchronization.
    std::unique_ptr<CudaEvent> tmp_event;
    if (event == nullptr) {
        tmp_event = std::make_unique<CudaEvent>();
        event = tmp_event.get();
    }

    // Let all downstreams wait on all upstreams.
    for (cuda::stream_ref const& upstream : upstreams) {
        event->record(upstream);
        for (cuda::stream_ref const& downstream : downstreams) {
            if (upstream.get() != downstream.get()) {
                event->stream_wait(downstream);
            }
        }
    }
}

/**
 * @brief Make a downstream CUDA stream wait on an upstream CUDA stream.
 *
 * This call is asynchronous with respect to the host thread; no host-side
 * blocking occurs.
 *
 * Equivalent to calling the range overload with one upstream and one downstream.
 *
 * @param downstream Stream that must not run ahead.
 * @param upstream Stream whose already-enqueued work must complete first.
 * @param event Optional CUDA event used for synchronization. A unique event per
 * call is not required; the same event may be reused. If `nullptr`, a temporary
 * event is created internally to avoid the small overhead of constructing one
 * per call site.
 *
 * @note If @p downstream and @p upstream are identical, this function is a no-op.
 *
 * @see cuda_stream_join(Range1 const&, Range2 const&, CudaEvent*)
 */
inline void cuda_stream_join(
    cuda::stream_ref downstream, cuda::stream_ref upstream, CudaEvent* event = nullptr
) {
    return cuda_stream_join(
        std::views::single(downstream), std::views::single(upstream), event
    );
}

}  // namespace rapidsmpf
