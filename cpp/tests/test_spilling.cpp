/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

class SpillingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = rmm::cuda_stream_default;
    }

    std::shared_ptr<BufferResource> br;
    rmm::cuda_stream_view stream;
};

TEST_F(SpillingTest, SpillUnspillRoundtripPreservesDataAndMetadata) {
    std::vector<std::uint8_t> metadata{42, 99};
    std::vector<std::uint8_t> payload{10, 20, 30};

    // Create device input.
    std::vector<PackedData> input;
    input.push_back(create_packed_data(metadata, payload, stream, br.get()));

    // Device -> Device (moves data)
    auto on_gpu = unspill_partitions(std::move(input), br.get(), AllowOverbooking::YES);
    ASSERT_EQ(on_gpu.size(), 1);
    EXPECT_EQ(on_gpu[0].data->mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(*on_gpu[0].metadata, metadata);

    // Device -> Host
    auto back_on_host = spill_partitions(std::move(on_gpu), br.get());
    ASSERT_EQ(back_on_host.size(), 1);
    EXPECT_EQ(back_on_host[0].data->mem_type(), MemoryType::HOST);
    EXPECT_EQ(*back_on_host[0].metadata, metadata);

    // Check that contents match original
    auto res = br->reserve_or_fail(back_on_host[0].data->size, MemoryType::HOST);
    auto actual = br->move_to_host_buffer(std::move(back_on_host[0].data), res);
    EXPECT_EQ(actual->copy_to_uint8_vector(), payload);
}
