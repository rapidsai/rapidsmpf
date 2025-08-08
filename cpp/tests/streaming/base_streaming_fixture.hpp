/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>

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
        auto comm = std::make_shared<rapidsmpf::Single>(options);
        stream = cudf::get_default_stream();
        br = std::make_unique<rapidsmpf::BufferResource>(mr_cuda);
        ctx = std::make_unique<rapidsmpf::streaming::Context>(
            options, comm, stream, std::make_shared<rapidsmpf::BufferResource>(mr_cuda)
        );
    }

    rmm::mr::cuda_memory_resource mr_cuda;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::streaming::Context> ctx;
};
