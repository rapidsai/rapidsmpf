/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "../environment.hpp"
#include "rapidsmpf/memory/pinned_memory_resource.hpp"

extern Environment* GlobalEnvironment;

class BaseStreamingFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        SetUpWithThreads(1);  // default number of streaming threads
    }

    void TearDown() override {
        ctx.reset();
        br.reset();
    }

    void SetUpWithThreads(
        int num_streaming_threads,
        std::unordered_map<
            rapidsmpf::MemoryType,
            rapidsmpf::BufferResource::MemoryAvailable> memory_available = {}
    ) {
        // create a new options object, since we can not modify values in the global
        // options object
        auto env_vars = rapidsmpf::config::get_environment_variables();
        env_vars["num_streaming_threads"] = std::to_string(num_streaming_threads);
        rapidsmpf::config::Options options(std::move(env_vars));

        stream = cudf::get_default_stream();
        br = std::make_shared<rapidsmpf::BufferResource>(
            mr_cuda, rapidsmpf::PinnedMemoryResource::Disabled, memory_available
        );
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            std::move(options), GlobalEnvironment->comm_, br
        );
    }

    rmm::cuda_stream_view stream;
    rmm::mr::cuda_memory_resource mr_cuda;
    std::shared_ptr<rapidsmpf::BufferResource> br;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};
