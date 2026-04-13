/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/coll/halo_exchange.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;

namespace {

// Build a PackedData whose GPU buffer contains the given int32_t values.
PackedData make_packed_ints(
    std::shared_ptr<BufferResource> const& br, std::vector<int32_t> const& values
) {
    auto const nbytes = values.size() * sizeof(int32_t);
    auto stream = br->stream_pool().get_stream();
    auto buf = br->allocate(stream, br->reserve_or_fail(nbytes, MemoryType::DEVICE));
    buf->write_access([&](std::byte* ptr, rmm::cuda_stream_view s) {
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(ptr, values.data(), nbytes, cudaMemcpyHostToDevice, s.value())
        );
    });
    buf->latest_write_event().host_wait();
    // PackedData requires non-empty metadata when GPU data is non-empty.
    auto meta = std::make_unique<std::vector<uint8_t>>(1, std::uint8_t{0});
    return PackedData{std::move(meta), std::move(buf)};
}

// Copy a PackedData's GPU buffer to a host vector of int32_t.
std::vector<int32_t> unpack_ints(PackedData const& pd) {
    auto const nbytes = pd.data->size;
    RAPIDSMPF_EXPECTS(nbytes % sizeof(int32_t) == 0, "unexpected buffer size");
    std::vector<int32_t> out(nbytes / sizeof(int32_t));
    auto* ptr = pd.data->exclusive_data_access();
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        out.data(), ptr, nbytes, cudaMemcpyDeviceToHost, pd.data->stream().value()
    ));
    pd.data->stream().synchronize();
    pd.data->unlock();
    return out;
}

}  // namespace

class StreamingHaloExchange : public BaseStreamingFixture {};

// Two-rank round-trip: rank 0 sends rightward, rank 1 sends leftward.
TEST_F(StreamingHaloExchange, two_rank_exchange) {
    auto comm = GlobalEnvironment->comm_;
    auto rank = comm->rank();
    auto nranks = comm->nranks();
    if (nranks != 2) {
        GTEST_SKIP() << "Requires exactly 2 ranks";
    }

    std::optional<std::vector<int32_t>> from_left_result;
    std::optional<std::vector<int32_t>> from_right_result;

    auto task = [&]() -> coro::task<void> {
        streaming::HaloExchange he(ctx, comm, OpID{0});

        std::optional<PackedData> send_left = std::nullopt;
        std::optional<PackedData> send_right = std::nullopt;
        if (rank == 0) {
            send_right = make_packed_ints(ctx->br(), {10, 20, 30});
        } else {
            send_left = make_packed_ints(ctx->br(), {40, 50});
        }

        auto [fl, fr] = co_await he.exchange(std::move(send_left), std::move(send_right));
        if (fl) {
            from_left_result = unpack_ints(*fl);
        }
        if (fr) {
            from_right_result = unpack_ints(*fr);
        }
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task()));
    streaming::run_actor_network(std::move(pipeline));

    if (rank == 0) {
        EXPECT_FALSE(from_left_result.has_value());
        ASSERT_TRUE(from_right_result.has_value());
        EXPECT_THAT(*from_right_result, ::testing::ElementsAre(40, 50));
    } else {
        ASSERT_TRUE(from_left_result.has_value());
        EXPECT_FALSE(from_right_result.has_value());
        EXPECT_THAT(*from_left_result, ::testing::ElementsAre(10, 20, 30));
    }
}

// Each rank sends its rank value to both neighbors; boundary ranks receive nullopt
// for the absent direction.
TEST_F(StreamingHaloExchange, boundary_ranks) {
    auto comm = GlobalEnvironment->comm_;
    auto rank = comm->rank();
    auto nranks = comm->nranks();
    if (nranks < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    std::optional<std::vector<int32_t>> from_left_result;
    std::optional<std::vector<int32_t>> from_right_result;

    auto task = [&]() -> coro::task<void> {
        streaming::HaloExchange he(ctx, comm, OpID{0});

        std::optional<PackedData> send_left =
            (rank > 0) ? std::optional<PackedData>{make_packed_ints(ctx->br(), {rank})}
                       : std::nullopt;
        std::optional<PackedData> send_right =
            (rank < nranks - 1)
                ? std::optional<PackedData>{make_packed_ints(ctx->br(), {rank})}
                : std::nullopt;

        auto [fl, fr] = co_await he.exchange(std::move(send_left), std::move(send_right));
        if (fl) {
            from_left_result = unpack_ints(*fl);
        }
        if (fr) {
            from_right_result = unpack_ints(*fr);
        }
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task()));
    streaming::run_actor_network(std::move(pipeline));

    if (rank == 0) {
        EXPECT_FALSE(from_left_result.has_value());
    }
    if (rank == nranks - 1) {
        EXPECT_FALSE(from_right_result.has_value());
    }

    if (rank > 0) {
        ASSERT_TRUE(from_left_result.has_value());
        EXPECT_THAT(*from_left_result, ::testing::ElementsAre(rank - 1));
    }
    if (rank < nranks - 1) {
        ASSERT_TRUE(from_right_result.has_value());
        EXPECT_THAT(*from_right_result, ::testing::ElementsAre(rank + 1));
    }
}
