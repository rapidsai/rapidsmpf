/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <coro/latch.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/coll/sparse_alltoall.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;

namespace {

std::pair<std::vector<Rank>, std::vector<Rank>> ring_peers(
    std::shared_ptr<Communicator> const& comm
) {
    if (comm->nranks() == 1) {
        return {{}, {}};
    }
    auto const rank = comm->rank();
    auto const nranks = comm->nranks();
    return {
        {static_cast<Rank>((rank + nranks - 1) % nranks)},
        {static_cast<Rank>((rank + 1) % nranks)}
    };
}

PackedData make_payload(
    int metadata_value,
    int payload_value,
    MemoryType mem_type,
    std::shared_ptr<BufferResource> const& br
) {
    auto stream = br->stream_pool().get_stream();
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(int));
    std::memcpy(metadata->data(), &metadata_value, sizeof(int));

    auto data = br->allocate(stream, br->reserve_or_fail(sizeof(int), mem_type));
    data->write_access([&](std::byte* ptr, rmm::cuda_stream_view op_stream) {
        RAPIDSMPF_CUDA_TRY(
            rapidsmpf::cuda_memcpy_async(
                ptr, &payload_value, sizeof(payload_value), op_stream
            )
        );
    });
    return {std::move(metadata), std::move(data)};
}

int decode_metadata(PackedData const& packed_data) {
    int result = -1;
    std::memcpy(&result, packed_data.metadata->data(), sizeof(result));
    return result;
}

int decode_payload(PackedData const& packed_data) {
    int result = -1;
    RAPIDSMPF_CUDA_TRY(
        rapidsmpf::cuda_memcpy_async(
            &result,
            packed_data.data->data(),
            sizeof(result),
            packed_data.stream().value()
        )
    );
    packed_data.stream().synchronize();
    return result;
}

}  // namespace

class StreamingSparseAlltoall
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<std::tuple<int, MemoryType>> {
  public:
    void SetUp() override {
        BaseStreamingFixture::SetUpWithThreads(std::get<0>(GetParam()));
    }

    void TearDown() override {
        BaseStreamingFixture::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(
    StreamingSparseAlltoall,
    StreamingSparseAlltoall,
    ::testing::Combine(
        ::testing::Values(1, 4), ::testing::Values(MemoryType::HOST, MemoryType::DEVICE)
    ),
    [](testing::TestParamInfo<StreamingSparseAlltoall::ParamType> const& info) {
        return "nthreads_" + std::to_string(std::get<0>(info.param)) + "_mem_type_"
               + (std::get<1>(info.param) == MemoryType::HOST ? "HOST" : "DEVICE");
    }
);

TEST_P(StreamingSparseAlltoall, basic_ring_exchange) {
    auto const mem_type = std::get<1>(GetParam());
    auto comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);
    streaming::SparseAlltoall exchange(ctx, comm, OpID{0}, srcs, dsts);

    constexpr int n_inserts{16};
    std::vector<int> metadata_result;
    std::vector<int> payload_result;
    coro::event ready_to_extract{};

    auto insert_all = [&]() -> coro::task<void> {
        for (auto dst : dsts) {
            for (int insertion_id = 0; insertion_id < n_inserts; ++insertion_id) {
                exchange.insert(
                    dst,
                    make_payload(
                        comm->rank() * 100 + insertion_id,
                        comm->rank() * 1000 + insertion_id,
                        mem_type,
                        ctx->br()
                    )
                );
            }
        }
        co_await exchange.insert_finished();
        ready_to_extract.set(ctx->executor()->get());
    };

    auto extract_all = [&]() -> coro::task<void> {
        co_await ready_to_extract;
        if (srcs.empty()) {
            co_return;
        }
        auto data = exchange.extract(srcs.front());
        for (auto& pd : data) {
            metadata_result.push_back(decode_metadata(pd));
            payload_result.push_back(decode_payload(pd));
        }
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(insert_all()));
    pipeline.push_back(ctx->executor()->schedule(extract_all()));
    streaming::run_actor_network(std::move(pipeline));

    if (srcs.empty()) {
        EXPECT_TRUE(metadata_result.empty());
        EXPECT_TRUE(payload_result.empty());
        return;
    }
    ASSERT_EQ(metadata_result.size(), n_inserts);
    ASSERT_EQ(payload_result.size(), n_inserts);
    auto const src = srcs.front();
    for (int i = 0; i < n_inserts; ++i) {
        EXPECT_EQ(metadata_result[i], src * 100 + i);
        EXPECT_EQ(payload_result[i], src * 1000 + i);
    }
}

TEST_P(StreamingSparseAlltoall, simultaneous_different_tags) {
    auto comm = GlobalEnvironment->comm_;
    auto [srcs, dsts] = ring_peers(comm);

    std::array<std::vector<int>, 2> results{};

    auto task = [&](OpID tag, int offset) -> coro::task<void> {
        streaming::SparseAlltoall exchange(ctx, comm, tag, srcs, dsts);
        for (auto dst : dsts) {
            exchange.insert(
                dst,
                make_payload(
                    offset + comm->rank(),
                    offset * 10 + comm->rank(),
                    MemoryType::HOST,
                    ctx->br()
                )
            );
        }
        co_await exchange.insert_finished();
        if (!srcs.empty()) {
            auto data = exchange.extract(srcs.front());
            EXPECT_EQ(data.size(), 1);
            if (data.size() == 1) {
                results[static_cast<std::size_t>(tag)] = {
                    decode_metadata(data.front()), decode_payload(data.front())
                };
            }
        }
        co_return;
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task(OpID{0}, 10)));
    pipeline.push_back(ctx->executor()->schedule(task(OpID{1}, 20)));
    streaming::run_actor_network(std::move(pipeline));

    if (srcs.empty()) {
        EXPECT_TRUE(results[0].empty());
        EXPECT_TRUE(results[1].empty());
        return;
    }
    auto const src = srcs.front();
    EXPECT_THAT(results[0], ::testing::ElementsAre(10 + src, 100 + src));
    EXPECT_THAT(results[1], ::testing::ElementsAre(20 + src, 200 + src));
}
