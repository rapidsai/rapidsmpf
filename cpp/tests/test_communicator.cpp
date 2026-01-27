/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

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

    rapidsmpf::Communicator* comm;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
};

TEST_F(BaseCommunicatorTest, TagConstruction) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Tag validation independent of communicator type";
    }
    EXPECT_THROW(
        rapidsmpf::Tag(0, 1 << rapidsmpf::Tag::stage_id_bits), std::overflow_error
    );
    EXPECT_THROW(rapidsmpf::Tag(1 << rapidsmpf::Tag::op_id_bits, 0), std::overflow_error);
    EXPECT_NO_THROW(rapidsmpf::Tag(0, (1 << rapidsmpf::Tag::stage_id_bits) - 1));
    EXPECT_NO_THROW(
        rapidsmpf::Tag(
            (1 << rapidsmpf::Tag::op_id_bits) - 1,
            (1 << rapidsmpf::Tag::stage_id_bits) - 1
        )
    );
    EXPECT_THROW(rapidsmpf::Tag(0, -1), std::overflow_error);
    EXPECT_THROW(rapidsmpf::Tag(-1, 0), std::overflow_error);
}

class BasicCommunicatorTest
    : public BaseCommunicatorTest,
      public ::testing::WithParamInterface<rapidsmpf::MemoryType> {
  protected:
    rapidsmpf::MemoryType memory_type() {
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
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    constexpr int nelems{10};
    auto send_data_h = iota_vector<std::uint8_t>(nelems);
    auto send_buf = br->allocate(stream, br->reserve_or_fail(nelems, memory_type()));
    send_buf->write_access([&](std::byte* send_buf_data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            send_buf_data, send_data_h.data(), nelems, cudaMemcpyDefault, stream
        ));
    });
    send_buf->stream().synchronize();
    rapidsmpf::Tag tag{0, 0};
    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);

    auto recv_buf = br->allocate(stream, br->reserve_or_fail(nelems, memory_type()));
    recv_buf->stream().synchronize();
    auto recv_fut = comm->recv(comm->rank(), tag, std::move(recv_buf));
    std::ignore = comm->wait(std::move(send_fut));
    recv_buf = comm->wait(std::move(recv_fut));
    auto [host_reservation, host_ob] = br->reserve(
        rapidsmpf::MemoryType::HOST, nelems, rapidsmpf::AllowOverbooking::YES
    );
    auto recv_data_h = br->move_to_host_buffer(std::move(recv_buf), host_reservation);
    stream.synchronize();
    EXPECT_EQ(send_data_h, recv_data_h->copy_to_uint8_vector());
}
