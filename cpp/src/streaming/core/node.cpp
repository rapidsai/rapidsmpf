/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

void run_streaming_pipeline(std::vector<Node> nodes) {
    coro_results(coro::sync_wait(coro::when_all(std::move(nodes))));
}

}  // namespace rapidsmpf::streaming
