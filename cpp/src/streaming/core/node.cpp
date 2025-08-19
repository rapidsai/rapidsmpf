/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

void run_streaming_pipeline(std::vector<Node> nodes) {
    auto results = coro::sync_wait(coro::when_all(std::move(nodes)));
    for (auto& result : results) {
        // The node result itself is always `void` but we access it here to re-throw
        // possible unhandled exceptions.
        result.return_value();
    }
}

}  // namespace rapidsmpf::streaming
