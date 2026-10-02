/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>

#include <cuda/stream>

/**
 * @brief Enqueue a kernel that keeps @p stream busy for (at least) @p delay.
 *
 * The kernel runs a single thread that sleeps, so it occupies almost no GPU resources
 * and delays on different streams overlap.
 *
 * @param delay How long the kernel should run.
 * @param stream The stream to delay.
 */
void launch_delay_kernel(std::chrono::nanoseconds delay, cuda::stream_ref stream);
