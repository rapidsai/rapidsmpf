/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/system_info.hpp>

#include "environment.hpp"
#include "utils.hpp"

class BaseCommunicatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_.get();
        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = rapidsmpf::BufferResource::create(*mr);
        stream = cuda::stream_ref{cudaStreamLegacy};
    }

    void TearDown() override {}

    rapidsmpf::Communicator* comm;
    std::unique_ptr<rmm::mr::cuda_memory_resource> mr;
    cuda::stream_ref stream{cudaStreamLegacy};
    std::shared_ptr<rapidsmpf::BufferResource> br;
};

TEST_F(BaseCommunicatorTest, TagConstruction) {
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
    auto send_buf = br->make_buffer(stream, br->reserve_or_fail(nelems, memory_type()));
    send_buf->write_access([&](std::byte* send_buf_data, cuda::stream_ref stream) {
        RAPIDSMPF_CUDA_TRY(
            rapidsmpf::cuda_memcpy_async(
                send_buf_data, send_data_h.data(), nelems, stream
            )
        );
    });
    send_buf->stream().sync();
    rapidsmpf::Tag tag{0, 0};
    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);

    auto recv_buf = br->make_buffer(stream, br->reserve_or_fail(nelems, memory_type()));
    recv_buf->stream().sync();
    auto recv_fut = comm->recv(comm->rank(), tag, std::move(recv_buf));
    std::ignore = comm->wait(std::move(send_fut));
    recv_buf = comm->wait(std::move(recv_fut));
    auto [host_reservation, host_ob] = br->reserve(
        rapidsmpf::MemoryType::HOST, nelems, rapidsmpf::AllowOverbooking::YES
    );
    auto recv_data_h = br->move_to_host_buffer(std::move(recv_buf), host_reservation);
    stream.sync();
    EXPECT_EQ(send_data_h, recv_data_h->copy_to_uint8_vector());
}

namespace {
// MPI counts are `int`, so a message larger than INT_MAX bytes takes a different path
// (#78).
constexpr std::size_t large_message_nbytes =
    std::size_t{std::numeric_limits<int>::max()} + 12345;
// A multiple of 251 lets every staging chunk use the same pattern.
constexpr std::size_t pattern_chunk_nbytes = 251 * 16384;
constexpr auto large_message_timeout = std::chrono::minutes{5};

std::vector<std::uint8_t> large_message_pattern_chunk() {
    std::vector<std::uint8_t> data(pattern_chunk_nbytes);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint8_t>(i % 251);
    }
    return data;
}
}  // namespace

TEST_P(BasicCommunicatorTest, LargeMessageSendToSelf) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    if (comm->rank() != 0) {
        GTEST_SKIP() << "Large transfer runs on rank 0 only";
    }
    auto const nbytes = large_message_nbytes;
    constexpr std::size_t reserve_headroom = 64 * 1024 * 1024;
    auto const host_needed =
        reserve_headroom
        + (memory_type() == rapidsmpf::MemoryType::HOST ? 2 * nbytes : 0);
    if (rapidsmpf::get_total_host_memory() < host_needed) {
        GTEST_SKIP() << "Not enough host memory for the large-message test";
    }
    if (memory_type() == rapidsmpf::MemoryType::DEVICE) {
        auto const [free_device_memory, total_device_memory] =
            rmm::available_device_memory();
        (void)total_device_memory;
        if (free_device_memory < 2 * nbytes + reserve_headroom) {
            GTEST_SKIP() << "Not enough free device memory for the large-message test";
        }
    }
    if (br->memory_available(memory_type()) < static_cast<std::int64_t>(2 * nbytes)) {
        GTEST_SKIP() << "Buffer resource limit is too small for two large messages";
    }
    auto const pattern = large_message_pattern_chunk();
    auto send_buf = br->make_buffer(stream, br->reserve_or_fail(nbytes, memory_type()));
    send_buf->stream().sync();
    send_buf->write_access([&](std::byte* send_buf_data, cuda::stream_ref stream) {
        for (std::size_t offset = 0; offset < nbytes; offset += pattern.size()) {
            auto const length = std::min(pattern.size(), nbytes - offset);
            if (memory_type() == rapidsmpf::MemoryType::HOST) {
                std::copy_n(
                    pattern.data(),
                    length,
                    reinterpret_cast<std::uint8_t*>(send_buf_data) + offset
                );
            } else {
                RAPIDSMPF_CUDA_TRY(
                    rapidsmpf::cuda_memcpy_async(
                        send_buf_data + offset, pattern.data(), length, stream
                    )
                );
            }
        }
    });
    send_buf->stream().sync();
    rapidsmpf::Tag tag{0, 0};
    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);

    auto recv_buf = br->make_buffer(stream, br->reserve_or_fail(nbytes, memory_type()));
    recv_buf->stream().sync();
    auto recv_fut = comm->recv(comm->rank(), tag, std::move(recv_buf));
    auto const deadline = std::chrono::steady_clock::now() + large_message_timeout;
    while (!comm->test(send_fut)) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Large send timed out";
        std::this_thread::yield();
    }
    std::ignore = comm->wait(std::move(send_fut));
    while (!comm->test(recv_fut)) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "Large receive timed out";
        std::this_thread::yield();
    }
    recv_buf = comm->wait(std::move(recv_fut));
    std::vector<std::uint8_t> staging;
    if (memory_type() == rapidsmpf::MemoryType::DEVICE) {
        staging.resize(pattern.size());
    }
    for (std::size_t offset = 0; offset < nbytes; offset += pattern.size()) {
        auto const length = std::min(pattern.size(), nbytes - offset);
        auto const* received =
            reinterpret_cast<std::uint8_t const*>(recv_buf->data()) + offset;
        if (memory_type() == rapidsmpf::MemoryType::DEVICE) {
            RAPIDSMPF_CUDA_TRY(
                rapidsmpf::cuda_memcpy_async(staging.data(), received, length, stream)
            );
            stream.sync();
            received = staging.data();
        }
        auto const mismatch =
            std::mismatch(pattern.begin(), pattern.begin() + length, received);
        ASSERT_EQ(mismatch.first, pattern.begin() + length)
            << "Large-message data mismatch at byte "
            << offset + (mismatch.first - pattern.begin());
    }
}

TEST_F(BaseCommunicatorTest, LargeHostMessageRecvFrom) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    if (comm->rank() != 0) {
        GTEST_SKIP() << "Large transfer runs on rank 0 only";
    }
    if (rapidsmpf::get_total_host_memory() < 2 * large_message_nbytes + 64 * 1024 * 1024)
    {
        GTEST_SKIP() << "Not enough host memory for the large-message test";
    }
    auto send_data = std::make_unique<std::vector<std::uint8_t>>(large_message_nbytes);
    for (std::size_t i = 0; i < send_data->size(); ++i) {
        (*send_data)[i] = static_cast<std::uint8_t>(i % 251);
    }
    rapidsmpf::Tag tag{0, 1};
    auto send_fut = comm->send(std::move(send_data), comm->rank(), tag);
    auto received = comm->recv_from(comm->rank(), tag);
    auto const deadline = std::chrono::steady_clock::now() + large_message_timeout;
    while (!received) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "Large receive timed out";
        std::this_thread::yield();
        received = comm->recv_from(comm->rank(), tag);
    }
    while (!comm->test(send_fut)) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Large send timed out";
        std::this_thread::yield();
    }
    ASSERT_EQ(received->size(), large_message_nbytes);
    auto const pattern = large_message_pattern_chunk();
    for (std::size_t offset = 0; offset < received->size(); offset += pattern.size()) {
        auto const length = std::min(pattern.size(), received->size() - offset);
        auto const mismatch = std::mismatch(
            pattern.begin(), pattern.begin() + length, received->data() + offset
        );
        ASSERT_EQ(mismatch.first, pattern.begin() + length)
            << "Large-message data mismatch at byte "
            << offset + (mismatch.first - pattern.begin());
    }
}
