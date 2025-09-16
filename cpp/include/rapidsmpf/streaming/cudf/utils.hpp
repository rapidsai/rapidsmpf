/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::streaming::utils {

/**
 * @brief Make @p primary wait until all work currently enqueued on @p secondary
 * completes.
 *
 * Records @p event on @p secondary and inserts a wait for that event on @p primary.
 * This is fully asynchronous with respect to the host thread; no host-side blocking.
 *
 * @param primary The stream that must not run ahead.
 * @param secondary The stream whose already-enqueued work must complete first.
 * @param event The CUDA event to use for synchronization. The same event may be reused
 * across multiple calls; the caller does not need to provide an unique event each time.
 */
inline void sync_streams(
    rmm::cuda_stream_view primary,
    rmm::cuda_stream_view secondary,
    cudaEvent_t const& event
) {
    if (primary.value() != secondary.value()) {
        RAPIDSMPF_CUDA_TRY(cudaEventRecord(event, secondary));
        RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(primary, event));
    }
}

}  // namespace rapidsmpf::streaming::utils