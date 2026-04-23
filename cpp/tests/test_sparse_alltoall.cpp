/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "environment.hpp"

extern Environment* GlobalEnvironment;

namespace {

std::pair<std::vector<rapidsmpf::Rank>, std::vector<rapidsmpf::Rank>> ring_peers(
    std::shared_ptr<rapidsmpf::Communicator> const& comm
) {
    if (comm->nranks() == 1) {
        return {{}, {}};
    }
    auto const rank = comm->rank();
    auto const nranks = comm->nranks();
    return {
        {static_cast<rapidsmpf::Rank>((rank + nranks - 1) % nranks)},
        {static_cast<rapidsmpf::Rank>((rank + 1) % nranks)}
    };
}

void CUDART_CB sleep_on_stream(void* user_data) {
    auto* delay = static_cast<std::chrono::milliseconds*>(user_data);
    std::this_thread::sleep_for(*delay);
    delete delay;
}

rapidsmpf::PackedData make_payload(
    int metadata_value,
    int payload_value,
    rapidsmpf::MemoryType mem_type,
    rapidsmpf::BufferResource& br,
    rmm::cuda_stream_view stream
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(int));
    std::memcpy(metadata->data(), &metadata_value, sizeof(int));

    auto data = br.allocate(stream, br.reserve_or_fail(sizeof(int), mem_type));
    data->write_access([&](std::byte* ptr, rmm::cuda_stream_view op_stream) {
        if (mem_type == rapidsmpf::MemoryType::DEVICE) {
            RAPIDSMPF_CUDA_TRY(
                rapidsmpf::cuda_memcpy_async(
                    ptr, &payload_value, sizeof(payload_value), op_stream
                )
            );
        } else {
            std::memcpy(ptr, &payload_value, sizeof(payload_value));
        }
    });
    return {std::move(metadata), std::move(data)};
}

int decode_metadata(rapidsmpf::PackedData const& packed_data) {
    int result;
    std::memcpy(&result, packed_data.metadata->data(), sizeof(result));
    return result;
}

int decode_payload(rapidsmpf::PackedData const& packed_data) {
    int result = -1;
    if (packed_data.data->mem_type() == rapidsmpf::MemoryType::DEVICE) {
        RAPIDSMPF_CUDA_TRY(
            rapidsmpf::cuda_memcpy_async(
                &result,
                packed_data.data->data(),
                sizeof(result),
                packed_data.data->stream()
            )
        );
        packed_data.data->stream().synchronize();
    } else {
        std::memcpy(&result, packed_data.data->data(), sizeof(result));
    }
    return result;
}

}  // namespace

class SparseAlltoallTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<rapidsmpf::BufferResource>(rmm::mr::cuda_memory_resource{});
    }

    std::unique_ptr<rapidsmpf::BufferResource> br;
};

TEST_F(SparseAlltoallTest, validate_constructor) {
    auto const& comm = GlobalEnvironment->comm_;
    auto const self = comm->rank();
    auto const size = comm->nranks();
    EXPECT_THROW(
        std::ignore = rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {self}, {}),
        std::out_of_range
    );
    EXPECT_THROW(
        std::ignore = rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {size}, {}),
        std::out_of_range
    );
    EXPECT_THROW(
        std::ignore = rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {}, {self}),
        std::out_of_range
    );
    EXPECT_THROW(
        std::ignore = rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {}, {size}),
        std::out_of_range
    );
    if (comm->nranks() > 1) {
        auto peer = (self + 1) % size;
        EXPECT_THROW(
            std::ignore =
                rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {peer, peer}, {}),
            std::invalid_argument
        );
        EXPECT_THROW(
            std::ignore =
                rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), {}, {peer, peer}),
            std::invalid_argument
        );
    }
}

class SparseAlltoallMemoryTest
    : public SparseAlltoallTest,
      public ::testing::WithParamInterface<rapidsmpf::MemoryType> {};

INSTANTIATE_TEST_SUITE_P(
    SparseAlltoall,
    SparseAlltoallMemoryTest,
    ::testing::Values(rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE),
    [](auto const& info) {
        return info.param == rapidsmpf::MemoryType::HOST ? "host" : "device";
    }
);

TEST_P(SparseAlltoallMemoryTest, basic_ring_exchange) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    int callback_count{0};
    std::mutex mutex;
    std::condition_variable cv;
    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts, [&]() {
        {
            std::lock_guard lk{mutex};
            callback_count++;
        }
        cv.notify_one();
    });

    auto const mem_type = GetParam();
    if (!dsts.empty()) {
        auto const dst = dsts.front();
        for (int i = 0; i < 3; ++i) {
            exchange.insert(
                dst,
                make_payload(
                    comm->rank() * 10 + i,
                    comm->rank() * 100 + i,
                    mem_type,
                    *br,
                    br->stream_pool().get_stream()
                )
            );
        }
    }
    exchange.insert_finished();
    {
        std::unique_lock lk{mutex};
        cv.wait_for(lk, std::chrono::seconds{30}, [&]() { return callback_count > 0; });
    }
    EXPECT_EQ(callback_count, 1);
    if (!srcs.empty()) {
        auto received = exchange.extract(srcs.front());
        ASSERT_EQ(received.size(), 3);
        auto const src = srcs.front();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(decode_metadata(received[i]), src * 10 + i);
            EXPECT_EQ(decode_payload(received[i]), src * 100 + i);
        }
    }
}

TEST_F(SparseAlltoallTest, zero_message_edge_and_callback) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    int callback_count{0};
    std::mutex mutex;
    std::condition_variable cv;

    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts, [&]() {
        {
            std::lock_guard lk{mutex};
            callback_count++;
        }
        cv.notify_one();
    });
    exchange.insert_finished();
    {
        std::unique_lock lk{mutex};
        cv.wait_for(lk, std::chrono::seconds{30}, [&]() { return callback_count > 0; });
    }
    EXPECT_EQ(callback_count, 1);
    if (!srcs.empty()) {
        auto received = exchange.extract(srcs.front());
        EXPECT_TRUE(received.empty());
    }
}

TEST_F(SparseAlltoallTest, asymmetric_peer_sets) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() < 3) {
        GTEST_SKIP() << "requires at least 3 ranks";
    }

    std::vector<rapidsmpf::Rank> srcs;
    std::vector<rapidsmpf::Rank> dsts;
    switch (comm->rank()) {
    case 0:
        srcs = {2};
        dsts = {1, 2};
        break;
    case 1:
        srcs = {0};
        dsts = {2};
        break;
    case 2:
        srcs = {0, 1};
        dsts = {0};
        break;
    default:
        srcs = {};
        dsts = {};
        break;
    }

    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);
    for (auto dst : dsts) {
        exchange.insert(
            dst,
            make_payload(
                comm->rank() * 10 + dst,
                comm->rank() * 100 + dst,
                rapidsmpf::MemoryType::DEVICE,
                *br,
                br->stream_pool().get_stream()
            )
        );
    }

    exchange.insert_finished();
    if (!dsts.empty()) {
        EXPECT_THROW(
            exchange.insert(
                dsts.front(),
                make_payload(
                    1, 1, rapidsmpf::MemoryType::HOST, *br, br->stream_pool().get_stream()
                )
            ),
            std::logic_error
        );
    }
    exchange.wait(std::chrono::seconds{30});

    for (auto src : srcs) {
        auto received = exchange.extract(src);
        ASSERT_EQ(received.size(), 1);
        EXPECT_EQ(decode_metadata(received.front()), src * 10 + comm->rank());
        EXPECT_EQ(decode_payload(received.front()), src * 100 + comm->rank());
    }
    EXPECT_THROW(std::ignore = exchange.extract(comm->rank()), std::logic_error);
}

TEST_F(SparseAlltoallTest, ordered_by_sender_insertion_with_stream_reordering) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "requires at least 2 ranks";
    }

    auto [srcs, dsts] = ring_peers(comm);
    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);

    auto const delayed_stream = br->stream_pool().get_stream(1);
    auto const fast_stream = br->stream_pool().get_stream(2);
    RAPIDSMPF_CUDA_TRY(cudaLaunchHostFunc(
        delayed_stream.value(), sleep_on_stream, new std::chrono::milliseconds(100)
    ));

    exchange.insert(
        dsts.front(),
        make_payload(
            comm->rank() * 10,
            comm->rank() * 100,
            rapidsmpf::MemoryType::DEVICE,
            *br,
            delayed_stream
        )
    );
    exchange.insert(
        dsts.front(),
        make_payload(
            comm->rank() * 10 + 1,
            comm->rank() * 100 + 1,
            rapidsmpf::MemoryType::DEVICE,
            *br,
            fast_stream
        )
    );

    exchange.insert_finished();
    exchange.wait(std::chrono::seconds{30});

    auto received = exchange.extract(srcs.front());
    ASSERT_EQ(received.size(), 2);
    auto const src = srcs.front();
    EXPECT_EQ(decode_metadata(received[0]), src * 10);
    EXPECT_EQ(decode_payload(received[0]), src * 100);
    EXPECT_EQ(decode_metadata(received[1]), src * 10 + 1);
    EXPECT_EQ(decode_payload(received[1]), src * 100 + 1);
}

TEST_F(SparseAlltoallTest, concurrent_insertions) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "requires at least 2 ranks";
    }

    auto [srcs, dsts] = ring_peers(comm);
    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);

    constexpr int num_threads = 4;
    constexpr int messages_per_thread = 16;
    constexpr int total_messages = num_threads * messages_per_thread;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto const dst = dsts.front();
    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        threads.emplace_back([&, thread_idx]() {
            for (int i = 0; i < messages_per_thread; ++i) {
                auto const sequence = thread_idx * messages_per_thread + i;
                exchange.insert(
                    dst,
                    make_payload(
                        sequence,
                        comm->rank() * total_messages + sequence,
                        rapidsmpf::MemoryType::HOST,
                        *br,
                        br->stream_pool().get_stream()
                    )
                );
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    exchange.insert_finished();
    exchange.wait(std::chrono::seconds{30});

    auto received = exchange.extract(srcs.front());
    ASSERT_EQ(received.size(), total_messages);

    std::ranges::sort(received, std::less{}, &decode_metadata);

    auto const src = srcs.front();
    for (int i = 0; i < total_messages; ++i) {
        EXPECT_EQ(decode_metadata(received[i]), i);
        EXPECT_EQ(decode_payload(received[i]), src * total_messages + i);
    }
}

TEST_F(SparseAlltoallTest, invalid_usage) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);

    EXPECT_THROW(
        exchange.insert(
            comm->rank(),
            make_payload(
                1, 2, rapidsmpf::MemoryType::DEVICE, *br, br->stream_pool().get_stream()
            )
        ),
        std::logic_error
    );

    if (!srcs.empty()) {
        EXPECT_THROW(std::ignore = exchange.extract(srcs.front()), std::logic_error);
    }
    exchange.insert_finished();
    exchange.wait(std::chrono::seconds{30});
    if (!srcs.empty()) {
        EXPECT_THROW(std::ignore = exchange.extract(comm->rank()), std::logic_error);
        std::ignore = exchange.extract(srcs.front());
        EXPECT_TRUE(exchange.extract(srcs.front()).empty());
    }
}

TEST_F(SparseAlltoallTest, tag_reuse_after_wait) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);

    for (int iteration = 0; iteration < 2; ++iteration) {
        rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);
        if (!dsts.empty()) {
            exchange.insert(
                dsts.front(),
                make_payload(
                    iteration,
                    comm->rank() * 1000 + iteration,
                    rapidsmpf::MemoryType::DEVICE,
                    *br,
                    br->stream_pool().get_stream()
                )
            );
        }
        exchange.insert_finished();
        exchange.wait(std::chrono::seconds{30});
        if (!srcs.empty()) {
            auto received = exchange.extract(srcs.front());
            ASSERT_EQ(received.size(), 1);
            EXPECT_EQ(decode_metadata(received.front()), iteration);
            EXPECT_EQ(decode_payload(received.front()), srcs.front() * 1000 + iteration);
        }
    }
}

TEST_F(SparseAlltoallTest, simultaneous_different_tags) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "requires at least 2 ranks";
    }

    rapidsmpf::coll::SparseAlltoall exchange0(comm, 0, br.get(), srcs, dsts);
    rapidsmpf::coll::SparseAlltoall exchange1(comm, 1, br.get(), srcs, dsts);

    if (!dsts.empty()) {
        auto const dst = dsts.front();
        for (int i = 0; i < 2; ++i) {
            exchange0.insert(
                dst,
                make_payload(
                    100 + i,
                    comm->rank() * 1000 + 100 + i,
                    rapidsmpf::MemoryType::DEVICE,
                    *br,
                    br->stream_pool().get_stream()
                )
            );
            exchange1.insert(
                dst,
                make_payload(
                    200 + i,
                    comm->rank() * 1000 + 200 + i,
                    rapidsmpf::MemoryType::DEVICE,
                    *br,
                    br->stream_pool().get_stream()
                )
            );
        }
    }

    exchange0.insert_finished();
    exchange1.insert_finished();
    exchange0.wait(std::chrono::seconds{30});
    exchange1.wait(std::chrono::seconds{30});

    if (!srcs.empty()) {
        auto const src = srcs.front();
        auto received0 = exchange0.extract(src);
        auto received1 = exchange1.extract(src);
        ASSERT_EQ(received0.size(), 2);
        ASSERT_EQ(received1.size(), 2);
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(decode_metadata(received0[i]), 100 + i);
            EXPECT_EQ(decode_payload(received0[i]), src * 1000 + 100 + i);
            EXPECT_EQ(decode_metadata(received1[i]), 200 + i);
            EXPECT_EQ(decode_payload(received1[i]), src * 1000 + 200 + i);
        }
    }
}
