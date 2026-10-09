/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <rapidsmpf/error.hpp>

#include "delay_kernel.hpp"

namespace {

__device__ std::uint64_t global_timer_ns() {
    std::uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

__global__ void delay_kernel(std::uint64_t delay_ns) {
    auto const start = global_timer_ns();
    while (global_timer_ns() - start < delay_ns) {
        __nanosleep(1000);
    }
}

}  // namespace

void launch_delay_kernel(std::chrono::nanoseconds delay, cuda::stream_ref stream) {
    if (delay.count() <= 0) {
        return;
    }
    delay_kernel<<<1, 1, 0, stream.get()>>>(static_cast<std::uint64_t>(delay.count()));
    RAPIDSMPF_CUDA_TRY(cudaGetLastError());
}
