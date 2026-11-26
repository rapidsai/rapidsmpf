/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include "environment.hpp"
#include "fake_communicator.hpp"
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

[[noreturn]] void fail_large_message_timeout(char const* message) {
    // A pending request can still reference a future's buffer, and some sends
    // cannot be cancelled. Do not unwind those futures on timeout.
    try {
        ADD_FAILURE() << message;
    } catch (...) {
        // GTest can be configured to throw when recording a failure.
    }
    std::fflush(nullptr);
    std::_Exit(EXIT_FAILURE);
}

std::vector<std::uint8_t> large_message_pattern_chunk() {
    std::vector<std::uint8_t> data(pattern_chunk_nbytes);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint8_t>(i % 251);
    }
    return data;
}
}  // namespace

TEST_F(BaseCommunicatorTest, LargeMessageSendToSelf) {
    if (GlobalEnvironment->type() == TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Unsupported send to self";
    }
    if (comm->rank() != 0) {
        GTEST_SKIP() << "Large transfer runs on rank 0 only";
    }
    auto const nbytes = large_message_nbytes;
    auto const pattern = large_message_pattern_chunk();
    auto send_buf =
        br->make_buffer(stream, br->reserve_or_fail(nbytes, rapidsmpf::MemoryType::HOST));
    send_buf->stream().sync();
    send_buf->write_access([&](std::byte* send_buf_data, cuda::stream_ref) {
        auto* data = reinterpret_cast<std::uint8_t*>(send_buf_data);
        for (std::size_t offset = 0; offset < nbytes; offset += pattern.size()) {
            auto const length = std::min(pattern.size(), nbytes - offset);
            std::copy_n(pattern.data(), length, data + offset);
        }
    });
    send_buf->stream().sync();
    rapidsmpf::Tag tag{0, 0};
    auto send_fut = comm->send(std::move(send_buf), comm->rank(), tag);

    auto recv_buf =
        br->make_buffer(stream, br->reserve_or_fail(nbytes, rapidsmpf::MemoryType::HOST));
    recv_buf->stream().sync();
    auto recv_fut = comm->recv(comm->rank(), tag, std::move(recv_buf));
    auto const deadline = std::chrono::steady_clock::now() + large_message_timeout;
    while (!comm->test(send_fut)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail_large_message_timeout("Large send timed out");
        }
        std::this_thread::yield();
    }
    std::ignore = comm->wait(std::move(send_fut));
    while (!comm->test(recv_fut)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail_large_message_timeout("Large receive timed out");
        }
        std::this_thread::yield();
    }
    recv_buf = comm->wait(std::move(recv_fut));
    for (std::size_t offset = 0; offset < nbytes; offset += pattern.size()) {
        auto const length = std::min(pattern.size(), nbytes - offset);
        auto const* received =
            reinterpret_cast<std::uint8_t const*>(recv_buf->data()) + offset;
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
    auto send_data = std::make_unique<std::vector<std::uint8_t>>(large_message_nbytes);
    for (std::size_t i = 0; i < send_data->size(); ++i) {
        (*send_data)[i] = static_cast<std::uint8_t>(i % 251);
    }
    rapidsmpf::Tag tag{0, 1};
    auto send_fut = comm->send(std::move(send_data), comm->rank(), tag);
    auto received = comm->recv_from(comm->rank(), tag);
    auto const deadline = std::chrono::steady_clock::now() + large_message_timeout;
    while (!received) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail_large_message_timeout("Large receive timed out");
        }
        std::this_thread::yield();
        received = comm->recv_from(comm->rank(), tag);
    }
    while (!comm->test(send_fut)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail_large_message_timeout("Large send timed out");
        }
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

TEST(DroppedFutures, InFlightCommunicationAbortsTheCommunicator) {
    auto comm = std::make_shared<FakeCommunicator>(GlobalEnvironment->comm_);
    auto const state = comm->state();
    rapidsmpf::Tag const tag{0, 0};
    auto send = comm->send(std::make_unique<std::vector<std::uint8_t>>(1), 1, tag);
    auto recv =
        comm->recv_sync_host_data(1, tag, std::make_unique<std::vector<std::uint8_t>>(1));
    EXPECT_FALSE(comm->is_aborted());
    // Destroying the futures doesn't free their in-flight communication, but aborts the
    // communicator, since it may still match later messages.
    send.reset();
    recv.reset();
    EXPECT_EQ(state->live.load(), 2);
    EXPECT_TRUE(comm->is_aborted());
    EXPECT_THROW(
        std::ignore =
            comm->send(std::make_unique<std::vector<std::uint8_t>>(1), 1, {1, 0}),
        rapidsmpf::communicator_aborted
    );
    // Destroying the communicator cancels the receive and frees it. The send never
    // completes, so it is leaked.
    comm.reset();
    EXPECT_EQ(state->live.load(), 1);
    EXPECT_EQ(state->freed_in_flight.load(), 0);
}

TEST(DroppedFutures, CompletedCommunicationDoesNotAbort) {
    auto comm = std::make_shared<FakeCommunicator>(GlobalEnvironment->comm_);
    auto const state = comm->state();
    auto send = comm->send(std::make_unique<std::vector<std::uint8_t>>(1), 1, {0, 0});
    comm->complete_futures();
    send.reset();
    EXPECT_EQ(state->live.load(), 0);
    EXPECT_FALSE(comm->is_aborted());
}

TEST(DroppedFutures, StuckSendsShareOneDeadline) {
    auto comm = std::make_shared<FakeCommunicator>(GlobalEnvironment->comm_);
    auto const state = comm->state();
    // Waited on one at a time, destroying the communicator would take `num_sends` times
    // the deadline of about one second.
    constexpr int num_sends = 10;
    std::vector<std::unique_ptr<rapidsmpf::Communicator::Future>> sends;
    for (int i = 0; i < num_sends; ++i) {
        sends.push_back(
            comm->send(std::make_unique<std::vector<std::uint8_t>>(1), 1, {0, 0})
        );
    }
    sends.clear();
    auto const start = std::chrono::steady_clock::now();
    comm.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds{3});
    // The sends never completed, so they were leaked rather than freed.
    EXPECT_EQ(state->live.load(), num_sends);
    EXPECT_EQ(state->freed_in_flight.load(), 0);
}

TEST(DroppedFutures, InFlightReceiveAbortsTheCommunicator) {
    // A communicator of its own, since it gets aborted.
    auto comm = GlobalEnvironment->new_comm();
    if (comm == nullptr) {
        GTEST_SKIP() << "Requires a communicator with futures";
    }
    rapidsmpf::Tag const tag{0, 0};
    auto const src = (comm->rank() + 1) % comm->nranks();
    // Never matched, nothing is sent.
    auto future = comm->recv_sync_host_data(
        src, tag, std::make_unique<std::vector<std::uint8_t>>(1)
    );
    EXPECT_FALSE(comm->is_aborted());
    // Before any rank aborts the communicator, which is propagated to the others.
    GlobalEnvironment->barrier();
    future.reset();
    EXPECT_TRUE(comm->is_aborted());
    EXPECT_THROW(
        std::ignore = comm->recv_sync_host_data(
            src, tag, std::make_unique<std::vector<std::uint8_t>>(1)
        ),
        rapidsmpf::communicator_aborted
    );
    // Let the abort notifications of all ranks be delivered before any rank destroys
    // its communicator.
    GlobalEnvironment->barrier();
    // Destroying the communicator cancels the receive, so it doesn't wait for the
    // deadline of about one second.
    auto const start = std::chrono::steady_clock::now();
    comm.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds{900});
}

TEST(CommunicatorAbort, OperationsThrowOnceAborted) {
    auto comm = std::make_shared<rapidsmpf::Single>(
        std::make_shared<rapidsmpf::ProgressThread>(),
        rapidsmpf::Logger::from_options(GlobalEnvironment->options())
    );
    EXPECT_FALSE(comm->is_aborted());
    EXPECT_EQ(comm->recv_from(0, rapidsmpf::Tag{0, 0}), nullptr);

    comm->abort("injected failure");
    EXPECT_TRUE(comm->is_aborted());
    try {
        std::ignore = comm->recv_from(0, rapidsmpf::Tag{0, 0});
        FAIL() << "expected communicator_aborted";
    } catch (rapidsmpf::communicator_aborted const& e) {
        EXPECT_EQ(e.origin(), 0);
        EXPECT_EQ(e.reason(), "injected failure");
        EXPECT_STREQ(e.what(), "communicator aborted by rank 0: injected failure");
    }

    // Only the first abort is recorded.
    comm->abort("aborting again");
    EXPECT_THAT(
        [&] { std::ignore = comm->recv_any(rapidsmpf::Tag{0, 0}); },
        ::testing::ThrowsMessage<rapidsmpf::communicator_aborted>(
            ::testing::HasSubstr("injected failure")
        )
    );
}

TEST(CommunicatorAbort, IdleOperationsFail) {
    // On a single rank, unfinished operations have no communication pending, they must
    // fail nonetheless once the communicator is aborted.
    auto make_comm = [] {
        return std::make_shared<rapidsmpf::Single>(
            std::make_shared<rapidsmpf::ProgressThread>(),
            rapidsmpf::Logger::from_options(GlobalEnvironment->options())
        );
    };
    auto const throws_aborted = ::testing::Throws<rapidsmpf::communicator_aborted>();
    constexpr auto timeout = std::chrono::seconds{60};
    auto br =
        rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());
    {
        auto comm = make_comm();
        rapidsmpf::coll::AllGather allgather(comm, 0, br.get());
        comm->abort("injected failure");
        EXPECT_THAT(
            [&] {
                std::ignore = allgather.wait_and_extract(
                    rapidsmpf::coll::AllGather::Ordered::NO, timeout
                );
            },
            throws_aborted
        );
    }
    {
        auto comm = make_comm();
        rapidsmpf::shuffler::Shuffler shuffler(comm, 0, 1, br.get());
        comm->abort("injected failure");
        EXPECT_THAT([&] { shuffler.wait(timeout); }, throws_aborted);
    }
    {
        auto comm = make_comm();
        rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), {}, {});
        comm->abort("injected failure");
        EXPECT_THAT([&] { exchange.wait(timeout); }, throws_aborted);
    }
}

TEST(CommunicatorAbort, AbortPropagatesToAllRanks) {
    // Abort a dedicated communicator, not the one shared by the other tests.
    auto comm = GlobalEnvironment->new_comm();
    if (comm == nullptr || comm->nranks() < 2) {
        GTEST_SKIP() << "Requires a dedicated communicator with at least 2 ranks";
    }
    auto br =
        rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());
    rapidsmpf::shuffler::Shuffler shuffler(
        comm, 0, static_cast<rapidsmpf::shuffler::PartID>(comm->nranks()), br.get()
    );
    // Rank 0 never finishes the shuffle, so it only completes by aborting.
    if (comm->rank() == 0) {
        comm->abort("injected failure");
    } else {
        shuffler.insert_finished();
    }
    try {
        shuffler.wait(std::chrono::seconds{60});
        FAIL() << "expected communicator_aborted";
    } catch (rapidsmpf::communicator_aborted const& e) {
        EXPECT_EQ(e.origin(), 0);
        EXPECT_EQ(e.reason(), "injected failure");
    } catch (std::runtime_error const& e) {
        // Not notified: abort locally, so that the shuffler fails and can be destroyed
        // instead of hanging.
        ADD_FAILURE() << "the abort was not propagated: " << e.what();
        comm->abort("abort not propagated");
        EXPECT_THROW(
            shuffler.wait(std::chrono::seconds{60}), rapidsmpf::communicator_aborted
        );
    }
    EXPECT_TRUE(comm->is_aborted());
}

TEST(CommunicatorAbort, DroppedInFlightCommunicationAbortsAllRanks) {
    // Abort a dedicated communicator, not the one shared by the other tests.
    auto comm = GlobalEnvironment->new_comm();
    if (comm == nullptr || comm->nranks() < 2) {
        GTEST_SKIP() << "Requires a dedicated communicator with at least 2 ranks";
    }
    // Rank 0 drops a receive that is never matched, aborting the communicator locally.
    // Its next check notifies the other ranks.
    if (comm->rank() == 0) {
        std::ignore = comm->recv_sync_host_data(
            1, rapidsmpf::Tag{0, 0}, std::make_unique<std::vector<std::uint8_t>>(1)
        );
    }
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    try {
        while (std::chrono::steady_clock::now() < deadline) {
            comm->throw_if_aborted();
            std::this_thread::yield();
        }
        FAIL() << "the abort was not propagated";
    } catch (rapidsmpf::communicator_aborted const& e) {
        EXPECT_EQ(e.origin(), 0);
        EXPECT_THAT(e.reason(), ::testing::HasSubstr("in flight"));
    }
    GlobalEnvironment->barrier();
}

TEST(CommunicatorAbort, AbortAfterDroppedInFlightCommunicationNotifiesAllRanks) {
    // Abort a dedicated communicator, not the one shared by the other tests.
    auto comm = GlobalEnvironment->new_comm();
    if (comm == nullptr || comm->nranks() < 2) {
        GTEST_SKIP() << "Requires a dedicated communicator with at least 2 ranks";
    }
    // Rank 0 drops a receive that is never matched, aborting the communicator locally,
    // and then aborts it, as after a failed operation. It never checks the abort, which
    // would notify the other ranks too: `abort()` must notify them.
    if (comm->rank() == 0) {
        std::ignore = comm->recv_sync_host_data(
            1, rapidsmpf::Tag{0, 0}, std::make_unique<std::vector<std::uint8_t>>(1)
        );
        EXPECT_TRUE(comm->is_aborted());
        comm->abort("injected failure");
    } else {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        try {
            while (std::chrono::steady_clock::now() < deadline) {
                comm->throw_if_aborted();
                std::this_thread::yield();
            }
            FAIL() << "the abort was not propagated";
        } catch (rapidsmpf::communicator_aborted const& e) {
            EXPECT_EQ(e.origin(), 0);
            EXPECT_EQ(e.reason(), "injected failure");
        }
    }
    GlobalEnvironment->barrier();
}
