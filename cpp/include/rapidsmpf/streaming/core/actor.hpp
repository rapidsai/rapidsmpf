/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Alias for an actor in a streaming graph.
 *
 * Actors represent coroutine-based asynchronous operations used throughout the streaming
 * graph.
 */
using Actor = coro::task<void>;

/**
 * @brief Runs a list of actors concurrently and waits for all to complete.
 *
 * This function schedules each actor and blocks until all of them have finished
 * execution. Typically used to launch multiple producer/consumer coroutines in parallel.
 *
 * @param actors A vector of actors to run.
 */
void run_actor_network(std::vector<Actor> actors);

}  // namespace rapidsmpf::streaming
