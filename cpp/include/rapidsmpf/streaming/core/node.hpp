/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Alias for a node in a streaming pipeline.
 *
 * Node represent coroutine-based asynchronous operations used throughout the streaming
 * pipeline.
 */
using Node = coro::task<void>;

/**
 * @brief Runs a list of nodes concurrently and waits for all to complete.
 *
 * This function schedules each node and blocks until all of them have finished execution.
 * Typically used to launch multiple producer/consumer coroutines in parallel.
 *
 * @param nodes A vector of nodes to run.
 */
void run_streaming_pipeline(std::vector<Node> nodes);

}  // namespace rapidsmpf::streaming
