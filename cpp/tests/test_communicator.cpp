/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>

#include <cudf_test/base_fixture.hpp>

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

class CommunicatorTest : public cudf::test::BaseFixtureWithParam<size_t> {
  protected:
    void SetUp() override {
        if (GlobalEnvironment != nullptr) {
            comm = GlobalEnvironment->comm_.get();
        }
        br = std::make_unique<BufferResource>(mr());

        n_ops = GetParam();
    }

    size_t n_ops;

    Communicator* comm;
    std::unique_ptr<BufferResource> br;
};

INSTANTIATE_TEST_CASE_P(
    CommunicatorTest,
    CommunicatorTest,
    ::testing::Values(1, 10),
    [](const testing::TestParamInfo<size_t>& info) {
        return "n_ops_" + std::to_string(info.param);
    }
);

// Test for the new multi-destination send method
TEST_P(CommunicatorTest, MultiDestinationSend) {
    if (GlobalEnvironment->type() != TestEnvironmentType::MPI) {
        GTEST_SKIP() << "Non-MPI communicators do not support multi-destination send";
    }

    auto data = iota_vector<uint8_t>(10);

    auto const all_ranks = iota_vector<Rank>(comm->nranks());

    // for each operation, each rank sends to every other rank, using the op iteration as
    // the op and rank as the stage of the tag
    std::vector<std::unique_ptr<Communicator::BatchFuture>> send_futures;
    std::vector<std::unique_ptr<Communicator::Future>> recv_futures;
    for (OpID op = 0; op < n_ops; ++op) {
        Rank this_rank = comm->rank();
        std::unordered_set<Rank> receivers(all_ranks.begin(), all_ranks.end());

        // every ranks receives from the other ranks. Post all receives first.
        for (Rank sender : all_ranks) {
            auto recv_buf = br->move(std::make_unique<decltype(data)>(data.size(), 0));
            recv_futures.emplace_back(comm->recv(
                sender, Tag{op, static_cast<StageID>(sender)}, std::move(recv_buf)
            ));
        }

        // Post batch send
        auto send_buf = br->move(std::make_unique<decltype(data)>(data));
        send_futures.emplace_back(comm->send(
            std::move(send_buf), receivers, Tag{op, static_cast<StageID>(this_rank)}
        ));
    }

    // wait for all sends to complete
    while (!std::all_of(send_futures.begin(), send_futures.end(), [&](auto& future) {
        return comm->test_batch(*future);
    }))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // wait for all receives to complete
    while (!recv_futures.empty()) {
        auto finished = comm->test_some(recv_futures);
        for (auto&& f : finished) {
            auto recv_data = comm->get_gpu_data(std::move(f));
            EXPECT_EQ(data.size(), recv_data->size);
            EXPECT_EQ(0, std::memcmp(data.data(), recv_data->data(), data.size()));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}
