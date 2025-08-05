/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <ranges>

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
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    constexpr int nelems{10};
    auto send_data_h = iota_vector<std::uint8_t>(nelems);
    auto [reservation, ob] = br->reserve(memory_type(), 2 * send_data_h.size(), true);
    auto send_buf = br->move(
        memory_type(),
        br->move(std::make_unique<std::vector<uint8_t>>(send_data_h)),
        stream,
        reservation
    );
    stream.synchronize();
    rapidsmpf::Tag tag{0, 0};

    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);
    auto recv_fut = comm->recv(
        comm->rank(),
        tag,
        br->allocate(memory_type(), send_data_h.size(), stream, reservation)
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

class CommunicatorTest
    : public BaseCommunicatorTest,
      public ::testing::WithParamInterface<std::tuple<rapidsmpf::MemoryType, size_t>> {
  protected:
    void SetUp() override {
        BaseCommunicatorTest::SetUp();
        std::tie(mem_type, n_ops) = GetParam();
    }

    rapidsmpf::MemoryType memory_type() override {
        return mem_type;
    }

    auto make_buffer(size_t size) {
        if (mem_type == rapidsmpf::MemoryType::HOST) {
            return br->move(std::make_unique<std::vector<uint8_t>>(size));
        } else {
            return br->move(std::make_unique<rmm::device_buffer>(size, stream), stream);
        }
    }

    auto copy_to_buffer(void* src, size_t size, rapidsmpf::Buffer& buf) {
        if (mem_type == rapidsmpf::MemoryType::HOST) {
            std::memcpy(buf.data(), src, size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(buf.data(), src, size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    auto copy_from_buffer(rapidsmpf::Buffer& buf, void* dst, size_t size) {
        if (mem_type == rapidsmpf::MemoryType::HOST) {
            std::memcpy(dst, buf.data(), size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(dst, buf.data(), size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    std::string vec_to_string(std::string const& prefix, std::vector<int> const& vec) {
        std::stringstream ss;
        ss << prefix << " ";
        for (auto&& v : vec) {
            ss << v << " ";
        }
        return ss.str();
    }

    rapidsmpf::MemoryType mem_type;
    rapidsmpf::OpID n_ops;
};

INSTANTIATE_TEST_SUITE_P(
    CommunicatorTest,
    CommunicatorTest,
    testing::Combine(
        testing::Values(rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE),
        testing::Values(1, 3)
    ),
    [](const testing::TestParamInfo<std::tuple<rapidsmpf::MemoryType, size_t>>& info) {
        return std::string{"memory_type_"}
               + (std::get<0>(info.param) == rapidsmpf::MemoryType::HOST ? "HOST"
                                                                         : "DEVICE")
               + "_n_ops_" + std::to_string(std::get<1>(info.param));
    }
);

// Test every rank sends a buffer to all other ranks (including itself)
TEST_P(CommunicatorTest, MultiDestinationSend) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "SINGLE communicator does not support multi-destination send";
    }

    constexpr size_t n_elems = 5;  // number of int elements to send
    auto all_ranks = iota_vector<rapidsmpf::Rank>(comm->nranks());

    // send data is arranged as follows:
    // | op 0                     | op 1                             | ... |
    // | rank 0  |  rank 1  | ... | rank 0            | rank 1 | ... |
    // | 0...n-1 | n...2n-1 | ... | n_ranks * n..     |...

    // for each operation, each rank sends to every other rank, using the op iteration as
    // the op and rank as the stage of the tag
    std::vector<std::unique_ptr<rapidsmpf::Communicator::Future>> send_futures;
    std::vector<std::unique_ptr<rapidsmpf::Communicator::Future>> recv_futures;
    for (rapidsmpf::OpID op = 0; op < n_ops; ++op) {
        // every ranks receives from the other ranks. Post all receives first.
        for (rapidsmpf::Rank sender : all_ranks) {
            auto recv_buf = make_buffer(n_elems * sizeof(int));
            recv_futures.emplace_back(comm->recv(
                sender,
                rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(sender)},
                std::move(recv_buf)
            ));
        }

        rapidsmpf::Rank this_rank = comm->rank();
        auto send_data =
            iota_vector<int>(n_elems, n_elems * (op * comm->nranks() + this_rank));

        // Post batch send
        auto send_buf = make_buffer(n_elems * sizeof(int));
        copy_to_buffer(send_data.data(), n_elems * sizeof(int), *send_buf);
        send_futures.emplace_back(comm->send(
            std::move(send_buf),
            all_ranks,
            rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(this_rank)}
        ));
    }

    // wait for all sends to complete
    for (auto& future : send_futures) {
        std::ignore = comm->wait(std::move(future));
    }

    // wait for all receives to complete
    std::vector<int> recv_data(n_elems * comm->nranks() * n_ops);
    size_t offset = 0;
    while (!recv_futures.empty()) {
        auto finished = comm->test_some(recv_futures);
        for (auto&& f : finished) {
            auto recv_buf = comm->get_gpu_data(std::move(f));
            copy_from_buffer(*recv_buf, recv_data.data() + offset, n_elems * sizeof(int));
            SCOPED_TRACE(
                vec_to_string(" offset: " + std::to_string(offset) + ":", recv_data)
            );
            offset += n_elems;
        }
    }
    EXPECT_EQ(offset, n_elems * comm->nranks() * n_ops);

    // sort recv data
    std::ranges::sort(recv_data);

    // check if the recv data is sorted
    SCOPED_TRACE(vec_to_string("recv_data:", recv_data));
    EXPECT_EQ(recv_data, iota_vector<int>(n_elems * comm->nranks() * n_ops));
}

// Test test_some with a mix of singleton and multi-req futures
TEST_P(CommunicatorTest, TestSomeMixedFutures) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "SINGLE communicator does not support communication";
    }

    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks, but only " << comm->nranks()
                     << " available";
    }

    constexpr int n_elems = 3;  // number of int elements to send

    auto singleton_ops = std::views::iota(rapidsmpf::OpID{0}, n_ops);
    auto multi_ops = std::views::iota(n_ops, rapidsmpf::OpID(n_ops * 2));

    auto all_ranks = iota_vector<rapidsmpf::Rank>(comm->nranks());
    rapidsmpf::Rank this_rank = comm->rank();

    // Create a vector of futures with mixed types:
    // - Singleton futures: individual sends and receives
    // - Multi-req futures: multi-destination sends
    std::vector<std::unique_ptr<rapidsmpf::Communicator::Future>> mixed_futures;
    std::vector<std::unique_ptr<rapidsmpf::Communicator::Future>> recv_futures;

    // Add some singleton futures (individual sends)
    for (rapidsmpf::OpID op : singleton_ops) {
        auto send_data = iota_vector<int>(n_elems, n_elems * (op * 10 + this_rank));
        auto send_buf = make_buffer(n_elems * sizeof(int));
        copy_to_buffer(send_data.data(), n_elems * sizeof(int), *send_buf);

        // Send to next rank (wrapping around)
        rapidsmpf::Rank dest_rank = (this_rank + 1) % comm->nranks();
        mixed_futures.emplace_back(comm->send(
            std::move(send_buf),
            dest_rank,
            rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(this_rank)}
        ));
    }

    // Add some multi-req futures (multi-destination sends)
    for (rapidsmpf::OpID op : multi_ops) {
        auto send_data = iota_vector<int>(n_elems, n_elems * (op * 10 + this_rank));
        auto send_buf = make_buffer(n_elems * sizeof(int));
        copy_to_buffer(send_data.data(), n_elems * sizeof(int), *send_buf);

        // Send to all ranks
        mixed_futures.emplace_back(comm->send(
            std::move(send_buf),
            all_ranks,
            rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(this_rank)}
        ));
    }

    std::vector<int> exp_data(n_elems * n_ops + n_elems * comm->nranks() * n_ops);
    size_t exp_offset = 0;
    // post receives
    // singleton receives
    for (rapidsmpf::OpID op : singleton_ops) {
        auto recv_buf = make_buffer(n_elems * sizeof(int));
        // Receive from previous rank (wrapping around)
        rapidsmpf::Rank src_rank = (this_rank + comm->nranks() - 1) % comm->nranks();
        recv_futures.emplace_back(comm->recv(
            src_rank,
            rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(src_rank)},
            std::move(recv_buf)
        ));
        for (auto&& i : std::views::iota(0, n_elems)) {
            exp_data[exp_offset++] = n_elems * (op * 10 + src_rank) + i;
        }
    }

    // multi-destination receives
    std::vector<int> exp_multi_data(n_elems * comm->nranks() * n_ops);
    for (rapidsmpf::OpID op : multi_ops) {
        for (rapidsmpf::Rank sender : all_ranks) {
            auto recv_buf = make_buffer(n_elems * sizeof(int));
            recv_futures.emplace_back(comm->recv(
                sender,
                rapidsmpf::Tag{op, static_cast<rapidsmpf::StageID>(sender)},
                std::move(recv_buf)
            ));
            for (auto&& i : std::views::iota(0, n_elems)) {
                exp_data[exp_offset++] = n_elems * (op * 10 + sender) + i;
            }
        }
    }

    // Test test_some multiple times until all futures are completed
    size_t total_completed = 0;
    while (!mixed_futures.empty()) {
        auto completed = comm->test_some(mixed_futures);
        total_completed += completed.size();
    }
    EXPECT_EQ(n_ops * 2, total_completed);

    // test_some on recv_futures
    std::vector<int> recv_data(exp_data.size());
    size_t recv_offset = 0;
    while (!recv_futures.empty()) {
        auto completed = comm->test_some(recv_futures);
        total_completed += completed.size();

        for (auto&& f : completed) {
            auto recv_buf = comm->get_gpu_data(std::move(f));
            copy_from_buffer(
                *recv_buf, recv_data.data() + recv_offset, n_elems * sizeof(int)
            );
            recv_offset += n_elems;
        }
    }

    std::ranges::sort(recv_data);
    std::ranges::sort(exp_data);

    SCOPED_TRACE(vec_to_string("exp_data:", exp_data));
    SCOPED_TRACE(vec_to_string("recv_data:", recv_data));
    EXPECT_EQ(exp_data, recv_data);
}
