/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

void run_streaming_pipeline(std::vector<Node> nodes) {
    auto results = coro::sync_wait(coro::when_all(std::move(nodes)));
    for (auto&& result : results) {
        // If a node results in an unhandled_exception, it is re-thrown here.
        result.return_value();
    }
}

}  // namespace rapidsmpf::streaming
