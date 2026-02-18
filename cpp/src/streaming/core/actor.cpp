/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>

namespace rapidsmpf::streaming {

void run_actor_graph(std::vector<Actor> actors) {
    coro_results(coro::sync_wait(coro::when_all(std::move(actors))));
}

}  // namespace rapidsmpf::streaming
