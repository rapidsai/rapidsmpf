/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

#include "environment.hpp"
#include "utils.hpp"

class BaseCommunicatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_.get();
        mr = std::unique_ptr<rmm::mr::device_memory_resource>(
            new rmm::mr::cuda_memory_resource{}
        );
        br = std::make_unique<rapidsmpf::BufferResource>(mr.get());
        stream = rmm::cuda_stream_default;
        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    virtual rapidsmpf::MemoryType memory_type() = 0;

    rapidsmpf::Communicator* comm;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
};

class BasicCommunicatorTest
    : public BaseCommunicatorTest,
      public ::testing::WithParamInterface<rapidsmpf::MemoryType> {
  protected:
    rapidsmpf::MemoryType memory_type() override {
        return GetParam();
    }
};

INSTANTIATE_TEST_SUITE_P(
    BasicCommunicatorTest,
    BasicCommunicatorTest,
    testing::Values(rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE),
    [](const testing::TestParamInfo<rapidsmpf::MemoryType>& info) {
        return (
            info.param == rapidsmpf::MemoryType::HOST ? "MemoryType_HOST"
                                                      : "MemoryType_DEVICE"
        );
    }
);

TEST_P(BasicCommunicatorTest, SendToSelf) {
    if (GlobalEnvironment->type() != TestEnvironmentType::MPI) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    constexpr int nelems{10};
    auto send_data_h = iota_vector<std::uint8_t>(nelems);
    auto reservation = br->reserve(memory_type(), 2 * send_data_h.size(), true);
    auto send_buf = rapidsmpf::BufferResource::move(
        rapidsmpf::BufferResource::move(
            std::make_unique<decltype(send_data_h)>(send_data_h)
        ),
        stream,
        reservation.first
    );
    stream.synchronize();
    rapidsmpf::Tag tag{0, 0};

    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);
    auto recv_fut = comm->recv(
        comm->rank(), tag, br->allocate(send_data_h.size(), stream, reservation.first)
    );
    auto recv_buf = comm->wait(std::move(recv_fut));
    std::ignore = comm->wait(std::move(send_fut));
    auto host_reservation =
        br->reserve(rapidsmpf::MemoryType::HOST, send_data_h.size(), true);
    auto recv_data_h = rapidsmpf::BufferResource::move_to_host_vector(
        std::move(recv_buf), stream, host_reservation.first
    );
    stream.synchronize();
    EXPECT_EQ(send_data_h, *recv_data_h);
}
