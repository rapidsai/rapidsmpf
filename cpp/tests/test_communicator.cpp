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

// Base class for all communicator tests which
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

    virtual MemoryType memory_type() = 0;

    auto make_buffer(size_t size) {
        if (memory_type() == MemoryType::HOST) {
            return br->move(std::make_unique<std::vector<uint8_t>>(size));
        } else {
            return br->move(std::make_unique<rmm::device_buffer>(size, stream), stream);
        }
    }

    auto copy_to_buffer(void* src, size_t size, Buffer& buf) {
        if (memory_type() == MemoryType::HOST) {
            std::memcpy(buf.data(), src, size);
        } else {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(buf.data(), src, size, cudaMemcpyDefault, stream)
            );
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
        }
    }

    auto copy_from_buffer(Buffer& buf, void* dst, size_t size) {
        if (memory_type() == MemoryType::HOST) {
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
                              public testing::WithParamInterface<MemoryType> {
  protected:
    MemoryType memory_type() override {
        return GetParam();
    }
};

INSTANTIATE_TEST_CASE_P(
    BasicCommunicatorTest,
    BasicCommunicatorTest,
    testing::Values(MemoryType::HOST, MemoryType::DEVICE),
    [](const testing::TestParamInfo<MemoryType>& info) {
        return "memory_type_" + std::to_string(static_cast<int>(info.param));
    }
);

// Test send to self
TEST_P(BasicCommunicatorTest, SendToSelf) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "SINGLE communicator does not support send to self";
    }

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
