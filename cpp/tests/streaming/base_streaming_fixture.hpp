/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <filesystem>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include "../environment.hpp"

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
        std::optional<rapidsmpf::PinnedPoolProperties> pinned_pool_properties =
            rapidsmpf::PinnedMemoryDisabled,
        std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits = {},
        std::optional<rapidsmpf::Duration> periodic_spill_check =
            std::chrono::milliseconds{1},
        std::optional<std::filesystem::path> spill_directory = std::nullopt
    ) {
        // create a new options object, since we can not modify values in the global
        // options object
        auto env_vars = rapidsmpf::config::get_environment_variables();
        env_vars["num_streaming_threads"] = std::to_string(num_streaming_threads);
        rapidsmpf::config::Options options(std::move(env_vars));

        stream = cuda::stream_ref{cudaStreamLegacy};
        br = rapidsmpf::BufferResource::create(
            mr_cuda,
            pinned_pool_properties,
            std::move(memory_limits),
            periodic_spill_check,
            std::make_shared<rapidsmpf::StreamPool>(16),
            rapidsmpf::Statistics::disabled(),
            std::move(spill_directory)
        );
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            std::move(options), GlobalEnvironment->comm_->logger(), br
        );
    }

    cuda::stream_ref stream{cudaStreamLegacy};
    rmm::mr::cuda_memory_resource mr_cuda;
    std::shared_ptr<rapidsmpf::BufferResource> br;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};
