/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingErrorHandling = BaseStreamingFixture;

TEST_F(StreamingErrorHandling, UnhandledException) {
    std::vector<Node> nodes;

    nodes.push_back([](Context& ctx) -> Node {
        co_await ctx.executor()->schedule();
        throw std::runtime_error("unhandled_exception");
    }(*ctx));

    EXPECT_THROW(run_streaming_pipeline(std::move(nodes)), std::runtime_error);
}
