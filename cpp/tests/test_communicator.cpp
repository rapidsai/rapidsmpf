/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <tuple>

#include <gmock/gmock.h>

#include <cudf_test/base_fixture.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

class BaseCommunicatorTest : public cudf::test::BaseFixture {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_.get();
        br = std::make_unique<BufferResource>(mr());
        stream = rmm::cuda_stream_default;
        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    virtual rapidsmpf::MemoryType memory_type() = 0;

    auto make_buffer(size_t size) {
        if (memory_type() == rapidsmpf::MemoryType::HOST) {
            return br->move(std::make_unique<std::vector<uint8_t>>(size));
        } else {
            return br->move(std::make_unique<rmm::device_buffer>(size, stream), stream);
        }
    }

    auto copy_to_buffer(void* src, size_t size, Buffer& buf) {
        if (memory_type() == rapidsmpf::MemoryType::HOST) {
            std::memcpy(buf.data(), src, size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(buf.data(), src, size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    auto copy_from_buffer(Buffer& buf, void* dst, size_t size) {
        if (memory_type() == rapidsmpf::MemoryType::HOST) {
            std::memcpy(dst, buf.data(), size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(dst, buf.data(), size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    void log_vec(std::string const& prefix, auto const& vec) {
        std::stringstream ss;
        ss << prefix << " ";
        for (auto&& v : vec) {
            ss << static_cast<int>(v) << " ";
        }
        comm->logger().debug(ss.str());
    }

    Communicator* comm;
    rmm::cuda_stream_view stream;
    std::unique_ptr<BufferResource> br;
};

class BasicCommunicatorTest : public BaseCommunicatorTest,
                              public testing::WithParamInterface<rapidsmpf::MemoryType> {
  protected:
    rapidsmpf::MemoryType memory_type() override {
        return GetParam();
    }
};

INSTANTIATE_TEST_CASE_P(
    BasicCommunicatorTest,
    BasicCommunicatorTest,
    testing::Values(rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE),
    [](const testing::TestParamInfo<rapidsmpf::MemoryType>& info) {
        return "memory_type_" + std::to_string(static_cast<int>(info.param));
    }
);

// Test send to self
TEST_P(BasicCommunicatorTest, SendToSelf) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "SINGLE communicator does not support send to self";
    }

    // if (comm->rank() == 0) {
    //     GTEST_SKIP();
    // }

    constexpr size_t n_elems = 10;
    auto data = iota_vector<uint8_t>(n_elems, 0);
    constexpr Tag tag{0, 0};
    Rank self_rank = comm->rank();

    auto send_buf = make_buffer(n_elems);
    copy_to_buffer(data.data(), n_elems, *send_buf);

    // send data to self (ignore the return future)
    auto send_future = comm->send(std::move(send_buf), self_rank, tag);

    // receive data from self
    std::vector<std::unique_ptr<Communicator::Future>> recv_futures;
    auto recv_buf = make_buffer(n_elems);
    recv_futures.emplace_back(comm->recv(self_rank, tag, std::move(recv_buf)));

    while (!recv_futures.empty()) {
        auto finished = comm->test_some(recv_futures);

        if (finished.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        // there should only be one finished future
        auto recv_buf = comm->get_gpu_data(std::move(finished[0]));
        EXPECT_EQ(n_elems, recv_buf->size);
        std::vector<uint8_t> recv_data(n_elems);
        copy_from_buffer(*recv_buf, recv_data.data(), n_elems);
        EXPECT_EQ(data, recv_data);
        log_vec("recv_data:", recv_data);
    }
}

class CommunicatorTest
    : public BaseCommunicatorTest,
      public testing::WithParamInterface<std::tuple<rapidsmpf::MemoryType, size_t>> {
  protected:
    void SetUp() override {
        std::tie(mem_type, n_ops) = GetParam();
        BaseCommunicatorTest::SetUp();
    }

    rapidsmpf::MemoryType memory_type() override {
        return mem_type;
    }

    rapidsmpf::MemoryType mem_type;
    size_t n_ops;
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
    while (std::ranges::any_of(send_futures, [&](auto& future) {
        return !comm->test_batch(*future);
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
    std::ranges::sort(recv_data);

    // check if the recv data is sorted
    for (int i = 0; i < static_cast<int>(recv_data.size()); ++i) {
        EXPECT_EQ(i, recv_data[i]);
    }
    log_vec("recv_data:", recv_data);
}
