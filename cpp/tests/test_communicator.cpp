/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tuple>

#include <gmock/gmock.h>

#include <cudf_test/base_fixture.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

class CommunicatorTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<rapidsmpf::MemoryType, size_t>> {
  protected:
    void SetUp() override {
        if (GlobalEnvironment != nullptr) {
            comm = GlobalEnvironment->comm_.get();
        }
        br = std::make_unique<BufferResource>(mr());
        stream = rmm::cuda_stream_default;
        std::tie(memory_type, n_ops) = GetParam();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    auto make_buffer(size_t size) {
        if (memory_type == rapidsmpf::MemoryType::HOST) {
            return br->move(std::make_unique<std::vector<uint8_t>>(size));
        } else {
            return br->move(std::make_unique<rmm::device_buffer>(size, stream), stream);
        }
    }

    auto copy_to_buffer(void* src, size_t size, Buffer& buf) {
        if (memory_type == rapidsmpf::MemoryType::HOST) {
            std::memcpy(buf.data(), src, size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(buf.data(), src, size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    auto copy_from_buffer(Buffer& buf, void* dst, size_t size) {
        if (memory_type == rapidsmpf::MemoryType::HOST) {
            std::memcpy(dst, buf.data(), size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(dst, buf.data(), size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    rapidsmpf::MemoryType memory_type;
    size_t n_ops;

    Communicator* comm;
    rmm::cuda_stream_view stream;
    std::unique_ptr<BufferResource> br;
};

INSTANTIATE_TEST_CASE_P(
    CommunicatorTest,
    CommunicatorTest,
    testing::Combine(
        testing::Values(rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE),
        testing::Values(1, 3)
    ),
    [](const testing::TestParamInfo<std::tuple<rapidsmpf::MemoryType, size_t>>& info) {
        return "memory_type_" + std::to_string(static_cast<int>(std::get<0>(info.param)))
               + "_n_ops_" + std::to_string(std::get<1>(info.param));
    }
);

// Test every rank sends a buffer to all other ranks (including itself)
TEST_P(CommunicatorTest, MultiDestinationSend) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "SINGLE communicator does not support multi-destination send";
    }

    auto log_vec = [&](std::string const& prefix, std::vector<int> const& vec) {
        std::stringstream ss;
        ss << prefix << " ";
        for (auto&& v : vec) {
            ss << v << " ";
        }
        comm->logger().debug(ss.str());
    };

    constexpr size_t n_elems = 5;  // number of int elements to send
    auto const all_ranks = iota_vector<Rank>(comm->nranks());

    // send data is arranged as follows:
    // | op 0                     | op 1                             | ... |
    // | rank 0  |  rank 1  | ... | rank 0            | rank 1 | ... |
    // | 0...n-1 | n...2n-1 | ... | n_ranks * n..     |...

    // for each operation, each rank sends to every other rank, using the op iteration as
    // the op and rank as the stage of the tag
    std::vector<std::unique_ptr<Communicator::BatchFuture>> send_futures;
    std::vector<std::unique_ptr<Communicator::Future>> recv_futures;
    for (OpID op = 0; op < n_ops; ++op) {
        // every ranks receives from the other ranks. Post all receives first.
        for (Rank sender : all_ranks) {
            auto recv_buf = make_buffer(n_elems * sizeof(int));
            recv_futures.emplace_back(comm->recv(
                sender, Tag{op, static_cast<StageID>(sender)}, std::move(recv_buf)
            ));
        }

        Rank this_rank = comm->rank();
        auto send_data =
            iota_vector<int>(n_elems, n_elems * (op * comm->nranks() + this_rank));

        std::unordered_set<Rank> receivers(all_ranks.begin(), all_ranks.end());
        // Post batch send
        auto send_buf = make_buffer(n_elems * sizeof(int));
        copy_to_buffer(send_data.data(), n_elems * sizeof(int), *send_buf);
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
    std::vector<int> recv_data(n_elems * comm->nranks() * n_ops);
    size_t offset = 0;
    while (!recv_futures.empty()) {
        auto finished = comm->test_some(recv_futures);
        for (auto&& f : finished) {
            auto recv_buf = comm->get_gpu_data(std::move(f));
            copy_from_buffer(*recv_buf, recv_data.data() + offset, n_elems * sizeof(int));
            log_vec(" offset: " + std::to_string(offset) + ":", recv_data);
            offset += n_elems;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    EXPECT_EQ(offset, n_elems * comm->nranks() * n_ops);

    // sort recv data
    std::sort(recv_data.begin(), recv_data.end());

    // check if the recv data is sorted
    for (int i = 0; i < static_cast<int>(recv_data.size()); ++i) {
        EXPECT_EQ(i, recv_data[i]);
    }
    log_vec("recv_data:", recv_data);
}
