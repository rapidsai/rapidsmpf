/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

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

    virtual rapidsmpf::MemoryType memory_type() {
        return rapidsmpf::MemoryType::DEVICE;
    }

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
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    constexpr int nelems{10};
    auto send_data_h = iota_vector<std::uint8_t>(nelems);
    auto [reservation, ob] = br->reserve(memory_type(), 2 * send_data_h.size(), true);
    auto send_buf = br->move(
        br->move(std::make_unique<std::vector<uint8_t>>(send_data_h)), stream, reservation
    );
    stream.synchronize();
    rapidsmpf::Tag tag{0, 0};

    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);
    auto recv_fut = comm->recv(
        comm->rank(), tag, br->allocate(send_data_h.size(), stream, reservation)
    );
    std::ignore = comm->wait(std::move(send_fut));
    auto recv_buf = comm->wait(std::move(recv_fut));
    auto [host_reservation, host_ob] =
        br->reserve(rapidsmpf::MemoryType::HOST, send_data_h.size(), true);
    auto recv_data_h =
        br->move_to_host_vector(std::move(recv_buf), stream, host_reservation);
    stream.synchronize();
    EXPECT_EQ(send_data_h, *recv_data_h);
}

using namespace rapidsmpf;

TEST_F(BaseCommunicatorTest, UcxxTagSendRecvCb) {
    if (GlobalEnvironment->type() != TestEnvironmentType::UCXX) {
        GTEST_SKIP() << "UCXX only";
    }

    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }
    auto ucx_comm = static_cast<rapidsmpf::ucxx::UCXX*>(comm);
    Tag const ready_for_data_tag{0, 1};
    Tag const metadata_tag{0, 2};
    Tag const gpu_data_tag{0, 3};

    constexpr size_t nelems{8};
    constexpr shuffler::detail::ChunkID chunk_id{100};
    constexpr shuffler::PartID part_id{100};

    // Create dummy metadata and data
    auto metadata = iota_vector<std::uint8_t>(nelems, 100);  // Start from 100
    auto data = iota_vector<std::uint8_t>(nelems, 200);  // Start from 200

    // Create PackedData using the helper function
    auto packed_data = create_packed_data(metadata, data, stream, br.get());

    auto chunk = shuffler::detail::Chunk::from_packed_data(
        chunk_id, part_id, std::move(packed_data)
    );

    std::vector<std::unique_ptr<Communicator::Future>> futures;

    if (comm->rank() == 0) {
        // send metadata to rank 1
        auto serialized_metadata = chunk.serialize();

        // send metadata to rank 1
        futures.emplace_back(ucx_comm->send(
            std::move(serialized_metadata), Rank(1), metadata_tag, br.get()
        ));

        // recive ready for data from rank 1
        auto ready_for_data = std::make_unique<std::vector<uint8_t>>(
            shuffler::detail::ReadyForDataMessage::byte_size
        );

        futures.emplace_back(ucx_comm->recv_with_cb(
            Rank(1),
            ready_for_data_tag,
            br->move(std::move(ready_for_data)),
            [&](std::unique_ptr<Buffer> buf) {
                auto const& host_buf = br->move_to_host_vector(std::move(buf));
                EXPECT_EQ(
                    host_buf->size(), shuffler::detail::ReadyForDataMessage::byte_size
                );

                shuffler::detail::ChunkID cid;
                std::memcpy(&cid, host_buf->data(), sizeof(cid));
                EXPECT_EQ(cid, chunk_id);

                auto data_buf = chunk.release_data_buffer();
                data_buf->wait_for_ready();
                futures.emplace_back(
                    ucx_comm->send(std::move(data_buf), Rank(1), gpu_data_tag)
                );
            }
        ));
    } else if (comm->rank() == 1) {
        std::unique_ptr<std::vector<uint8_t>> recv_buf;
        Rank sender_rank;

        // wait for metadata from rank 0
        while (!recv_buf) {
            std::tie(recv_buf, sender_rank) = ucx_comm->recv_any(metadata_tag);
        }
        EXPECT_EQ(sender_rank, 0);
        auto chunk = shuffler::detail::Chunk::deserialize(*recv_buf);
        EXPECT_EQ(chunk.chunk_id(), chunk_id);

        // allocate data buffer
        auto [reservation, ob] =
            br->reserve(MemoryType::DEVICE, chunk.concat_data_size(), false);
        auto data_buf = br->allocate(chunk.concat_data_size(), stream, reservation);
        data_buf->wait_for_ready();

        // post recv for data from rank 0
        futures.emplace_back(ucx_comm->recv_with_cb(
            sender_rank,
            gpu_data_tag,
            std::move(data_buf),
            [data_size = chunk.concat_data_size()](std::unique_ptr<Buffer> buf) {
                EXPECT_EQ(data_size, buf->size);
            }
        ));

        // send ready for data to rank 0
        auto ready_for_data = std::make_unique<std::vector<uint8_t>>(
            shuffler::detail::ReadyForDataMessage::byte_size
        );
        std::memcpy(ready_for_data->data(), &chunk_id, sizeof(chunk_id));
        futures.emplace_back(ucx_comm->send(
            br->move(std::move(ready_for_data)), Rank(0), ready_for_data_tag
        ));
    }  // else do nothing

    while (!futures.empty()) {
        std::ignore = ucx_comm->test_some(futures);
    }
}
