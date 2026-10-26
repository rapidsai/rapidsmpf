/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

class SpillingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = cuda::stream_ref{cudaStreamLegacy};
    }

    std::shared_ptr<BufferResource> br;
    cuda::stream_ref stream{cudaStreamLegacy};
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

TEST_F(SpillingTest, EnsureMemoryTypeKeepsBufferInAcceptedType) {
    std::vector<std::uint8_t> metadata{1};
    std::vector<std::uint8_t> payload{10, 20, 30};
    auto data = create_packed_data(metadata, payload, stream, br.get()).data;
    auto const* ptr = data.get();
    auto ret = ensure_memory_type(std::move(data), ADDRESSABLE_MEMORY_TYPES, br.get());
    EXPECT_EQ(ret.get(), ptr);
}

TEST_F(SpillingTest, EnsureMemoryTypeRejectsInvalidArguments) {
    std::vector<std::uint8_t> metadata{1};
    std::vector<std::uint8_t> payload{10, 20, 30};
    EXPECT_THROW(
        std::ignore = ensure_memory_type(nullptr, ADDRESSABLE_MEMORY_TYPES, br.get()),
        std::invalid_argument
    );
    auto data = create_packed_data(metadata, payload, stream, br.get()).data;
    EXPECT_THROW(
        std::ignore =
            ensure_memory_type(std::move(data), std::span<MemoryType const>{}, br.get()),
        std::invalid_argument
    );
}

TEST_F(SpillingTest, EnsureMemoryTypeMovesIntoLastType) {
    TempDir temp_dir;
    auto disk_br = create_disk_br(temp_dir);
    std::vector<std::uint8_t> metadata{1};
    std::vector<std::uint8_t> payload{10, 20, 30};
    auto make_disk_data = [&] {
        return move_to_disk(
                   create_packed_data(metadata, payload, stream, disk_br.get()), *disk_br
        )
            .data;
    };

    // The last memory type is used even though device memory fits.
    auto ret =
        ensure_memory_type(make_disk_data(), ADDRESSABLE_MEMORY_TYPES, disk_br.get());
    EXPECT_EQ(ret->mem_type(), MemoryType::HOST);

    // Without host memory, the last memory type is overbooked.
    disk_br->set_memory_limit(MemoryType::HOST, 0);
    ret = ensure_memory_type(make_disk_data(), ADDRESSABLE_MEMORY_TYPES, disk_br.get());
    EXPECT_EQ(ret->mem_type(), MemoryType::HOST);
    ret->stream().sync();
    auto host_res =
        disk_br->reserve(MemoryType::HOST, payload.size(), AllowOverbooking::YES).first;
    EXPECT_EQ(
        disk_br->move_to_host_buffer(std::move(ret), host_res)->copy_to_uint8_vector(),
        payload
    );
}

TEST_F(SpillingTest, EnsureMemoryTypeThrowsWithoutOverbooking) {
    TempDir temp_dir;
    auto disk_br = create_disk_br(temp_dir, AllowOverbooking::NO);
    std::vector<std::uint8_t> metadata{1};
    std::vector<std::uint8_t> payload{10, 20, 30};
    auto data = move_to_disk(
                    create_packed_data(metadata, payload, stream, disk_br.get()), *disk_br
    )
                    .data;
    disk_br->set_memory_limit(MemoryType::HOST, 0);
    EXPECT_THROW(
        std::ignore =
            ensure_memory_type(std::move(data), ADDRESSABLE_MEMORY_TYPES, disk_br.get()),
        reservation_error
    );
}
