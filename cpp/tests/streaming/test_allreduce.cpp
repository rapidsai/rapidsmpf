/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include <vector>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allreduce.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;

namespace {

template <typename T>
std::unique_ptr<Buffer> make_buffer(
    std::shared_ptr<BufferResource> const& br,
    T const* data,
    std::size_t count,
    MemoryType mem_type
) {
    auto const nbytes = count * sizeof(T);
    auto stream = br->stream_pool().get_stream();
    auto reservation = br->reserve_or_fail(nbytes, mem_type);
    auto buffer = br->allocate(stream, std::move(reservation));
    buffer->write_access([&](std::byte* buf_data, rmm::cuda_stream_view s) {
        RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(buf_data, data, nbytes, s));
    });
    buffer->latest_write_event().host_wait();
    return buffer;
}

template <typename T>
std::vector<T> unpack_to_host(Buffer& buffer) {
    auto const nbytes = buffer.size;
    RAPIDSMPF_EXPECTS(
        nbytes % sizeof(T) == 0, "unpack_to_host: buffer size not a multiple of sizeof(T)"
    );
    auto const count = nbytes / sizeof(T);
    std::vector<T> out(count);
    auto* raw_ptr = buffer.exclusive_data_access();
    RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(out.data(), raw_ptr, nbytes, buffer.stream()));
    buffer.stream().synchronize();
    buffer.unlock();
    return out;
}

}  // namespace

class StreamingAllReduce : public BaseStreamingFixture,
                           public ::testing::WithParamInterface<int> {
  public:
    void SetUp() override {
        BaseStreamingFixture::SetUpWithThreads(4);
    }

    void TearDown() override {
        BaseStreamingFixture::TearDown();
    }
};

TEST_F(StreamingAllReduce, basic_sum) {
    auto comm = GlobalEnvironment->comm_;
    auto nranks = comm->nranks();
    auto rank = comm->rank();

    constexpr int n_elements{10};
    std::vector<int> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = rank + 1;  // rank 0 -> 1, rank 1 -> 2, etc.
    }

    std::vector<int> result;

    auto task = [&]() -> coro::task<void> {
        auto input =
            make_buffer<int>(ctx->br(), data.data(), data.size(), MemoryType::HOST);
        auto reservation = ctx->br()->reserve_or_fail(input->size, input->mem_type());
        auto output = ctx->br()->allocate(input->size, input->stream(), reservation);

        auto reduce_op = coll::detail::make_host_reduce_operator<int>(std::plus<int>{});

        streaming::AllReduce allreduce(
            ctx, comm, std::move(input), std::move(output), OpID{0}, std::move(reduce_op)
        );
        auto [in_buf, out_buf] = co_await allreduce.extract();
        result = unpack_to_host<int>(*out_buf);
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task()));
    streaming::run_actor_network(std::move(pipeline));

    int const expected = nranks * (nranks + 1) / 2;
    ASSERT_EQ(result.size(), n_elements);
    EXPECT_THAT(result, ::testing::Each(expected));
}

TEST_F(StreamingAllReduce, empty_buffer) {
    auto comm = GlobalEnvironment->comm_;

    std::vector<int> result;

    auto task = [&]() -> coro::task<void> {
        std::vector<int> empty;
        auto input = make_buffer<int>(ctx->br(), empty.data(), 0, MemoryType::HOST);
        auto reservation = ctx->br()->reserve_or_fail(input->size, input->mem_type());
        auto output = ctx->br()->allocate(input->size, input->stream(), reservation);

        auto reduce_op = coll::detail::make_host_reduce_operator<int>(std::plus<int>{});

        streaming::AllReduce allreduce(
            ctx, comm, std::move(input), std::move(output), OpID{0}, std::move(reduce_op)
        );
        auto [in_buf, out_buf] = co_await allreduce.extract();
        result = unpack_to_host<int>(*out_buf);
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task()));
    streaming::run_actor_network(std::move(pipeline));

    EXPECT_TRUE(result.empty());
}

TEST_F(StreamingAllReduce, concurrent_allreduces) {
    auto comm = GlobalEnvironment->comm_;
    auto nranks = comm->nranks();
    auto rank = comm->rank();

    constexpr int n_elements{5};

    std::vector<std::vector<int>> result(2);

    auto task = [&](int tag) -> coro::task<void> {
        std::vector<int> data(n_elements, (rank + 1) + tag * 10);
        auto input =
            make_buffer<int>(ctx->br(), data.data(), data.size(), MemoryType::HOST);
        auto reservation = ctx->br()->reserve_or_fail(input->size, input->mem_type());
        auto output = ctx->br()->allocate(input->size, input->stream(), reservation);

        auto reduce_op = coll::detail::make_host_reduce_operator<int>(std::plus<int>{});

        streaming::AllReduce allreduce(
            ctx, comm, std::move(input), std::move(output), tag, std::move(reduce_op)
        );
        auto [in_buf, out_buf] = co_await allreduce.extract();
        result[tag] = unpack_to_host<int>(*out_buf);
    };

    std::vector<coro::task<void>> pipeline{};
    pipeline.push_back(ctx->executor()->schedule(task(0)));
    pipeline.push_back(ctx->executor()->schedule(task(1)));
    streaming::run_actor_network(std::move(pipeline));

    int const expected_first = nranks * (nranks + 1) / 2;
    ASSERT_EQ(result[0].size(), n_elements);
    EXPECT_THAT(result[0], ::testing::Each(expected_first));

    int const expected_second = 10 * nranks + nranks * (nranks + 1) / 2;
    ASSERT_EQ(result[1].size(), n_elements);
    EXPECT_THAT(result[1], ::testing::Each(expected_second));
}
