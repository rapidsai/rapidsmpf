/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstring>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>

#include <rapidsmpf/coll/sparse_alltoall.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
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
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(int));
    std::memcpy(metadata->data(), &metadata_value, sizeof(int));

    auto data = br.allocate(stream, br.reserve_or_fail(sizeof(int), mem_type));
    data->write_access([&](std::byte* ptr, rmm::cuda_stream_view op_stream) {
        if (mem_type == rapidsmpf::MemoryType::DEVICE) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                ptr, &payload_value, sizeof(payload_value), cudaMemcpyDefault, op_stream
            ));
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
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            &result,
            packed_data.data->data(),
            sizeof(result),
            cudaMemcpyDefault,
            packed_data.data->stream()
        ));
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
        stream = cudf::get_default_stream();
        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = std::make_unique<rapidsmpf::BufferResource>(mr.get());
    }

    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

class SparseAlltoallValidateConstructorTest
    : public SparseAlltoallTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, std::vector<rapidsmpf::Rank>>> {};

auto sparse_alltoall_validate_constructor_name(
    ::testing::TestParamInfo<std::tuple<bool, std::vector<rapidsmpf::Rank>>> const& info
) -> std::string {
    auto const [invalid_srcs, invalid_peers] = info.param;
    auto const which = invalid_srcs ? "srcs" : "dsts";
    auto const peer = invalid_peers.front();
    auto const shape = invalid_peers.size() > 1 ? "duplicate"
                       : peer < 0               ? "negative"
                       : peer == 0              ? "self"
                                                : "out_of_range";
    return std::string{which} + "_" + shape;
}

INSTANTIATE_TEST_SUITE_P(
    SparseAlltoall,
    SparseAlltoallValidateConstructorTest,
    ::testing::Values(
        std::make_tuple(true, std::vector<rapidsmpf::Rank>{0}),
        std::make_tuple(false, std::vector<rapidsmpf::Rank>{0}),
        std::make_tuple(true, std::vector<rapidsmpf::Rank>{0, 0}),
        std::make_tuple(false, std::vector<rapidsmpf::Rank>{0, 0}),
        std::make_tuple(true, std::vector<rapidsmpf::Rank>{-1}),
        std::make_tuple(false, std::vector<rapidsmpf::Rank>{-1}),
        std::make_tuple(true, std::vector<rapidsmpf::Rank>{1 << 20}),
        std::make_tuple(false, std::vector<rapidsmpf::Rank>{1 << 20})
    ),
    sparse_alltoall_validate_constructor_name
);

TEST_P(SparseAlltoallValidateConstructorTest, validate_constructor) {
    auto const& comm = GlobalEnvironment->comm_;
    auto const [invalid_srcs, invalid_peers] = GetParam();
    auto const self = comm->rank();
    auto invalid = invalid_peers;
    if (invalid.size() == 1 && invalid.front() == 0) {
        invalid.front() = self;
    }
    auto const valid_other = (self + 1) % comm->nranks();
    auto const valid = self == valid_other ? std::vector<rapidsmpf::Rank>{}
                                           : std::vector<rapidsmpf::Rank>{valid_other};
    auto const srcs = invalid_srcs ? invalid : valid;
    auto const dsts = invalid_srcs ? valid : invalid;
    EXPECT_THROW(
        std::ignore = rapidsmpf::coll::SparseAlltoall(comm, 0, br.get(), srcs, dsts),
        std::logic_error
    );
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
                    comm->rank() * 10 + i, comm->rank() * 100 + i, mem_type, stream, *br
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
                stream,
                *br
            )
        );
    }

    exchange.insert_finished();
    exchange.wait(std::chrono::seconds{30});

    for (auto src : srcs) {
        auto received = exchange.extract(src);
        ASSERT_EQ(received.size(), 1);
        EXPECT_EQ(decode_metadata(received.front()), src * 10 + comm->rank());
        EXPECT_EQ(decode_payload(received.front()), src * 100 + comm->rank());
    }
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
            delayed_stream,
            *br
        )
    );
    exchange.insert(
        dsts.front(),
        make_payload(
            comm->rank() * 10 + 1,
            comm->rank() * 100 + 1,
            rapidsmpf::MemoryType::DEVICE,
            fast_stream,
            *br
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

TEST_F(SparseAlltoallTest, invalid_usage) {
    auto const& comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    rapidsmpf::coll::SparseAlltoall exchange(comm, 0, br.get(), srcs, dsts);

    EXPECT_THROW(
        exchange.insert(
            comm->rank(), make_payload(1, 2, rapidsmpf::MemoryType::DEVICE, stream, *br)
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
                    stream,
                    *br
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
                    stream,
                    *br
                )
            );
            exchange1.insert(
                dst,
                make_payload(
                    200 + i,
                    comm->rank() * 1000 + 200 + i,
                    rapidsmpf::MemoryType::DEVICE,
                    stream,
                    *br
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
