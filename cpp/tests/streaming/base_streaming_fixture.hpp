/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

class BaseStreamingFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        rapidsmpf::config::Options options{
            rapidsmpf::config::get_environment_variables()
        };
        stream = cudf::get_default_stream();
        br = std::make_unique<rapidsmpf::BufferResource>(mr_cuda);
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            options, std::make_shared<rapidsmpf::Single>(options), br.get()
        );
    }

    rmm::cuda_stream_view stream;
    rmm::mr::cuda_memory_resource mr_cuda;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};
