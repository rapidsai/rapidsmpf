/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "test_allreduce_custom_type.hpp"

using rapidsmpf::MemoryType;
using rapidsmpf::OpID;
using rapidsmpf::PackedData;
using rapidsmpf::coll::AllReduce;
using rapidsmpf::coll::ReduceKernel;
using rapidsmpf::coll::ReduceOp;

namespace {

template <typename T, ReduceOp Op>
struct AllReduceConfig {
    using value_type = T;
    static constexpr ReduceOp reduce_op_id = Op;
};

template <ReduceOp Op>
struct Combiner;

template <>
struct Combiner<ReduceOp::SUM> {
    template <typename T>
    static T apply(T a, T b) {
        return a + b;
    }
};

template <>
struct Combiner<ReduceOp::PROD> {
    template <typename T>
    static T apply(T a, T b) {
        return a * b;
    }
};

template <>
struct Combiner<ReduceOp::MIN> {
    template <typename T>
    static T apply(T a, T b) {
        return std::min(a, b);
    }
};

template <>
struct Combiner<ReduceOp::MAX> {
    template <typename T>
    static T apply(T a, T b) {
        return std::max(a, b);
    }
};

template <typename T>
T make_input_value(int rank, int insert_idx, int elem_idx) {
    // Simple positive pattern valid for all tested arithmetic types.
    auto base = static_cast<std::uint64_t>((rank + 1) * 100 + (insert_idx + 1) * 10);
    return static_cast<T>(base + elem_idx);
}

/**
 * @brief Helper to create a device-backed `PackedData` from host memory.
 */
template <typename T>
PackedData make_packed_from_host(
    rapidsmpf::BufferResource* br, T const* data, std::size_t count
) {
    auto const nbytes = count * sizeof(T);
    auto stream = br->stream_pool().get_stream();
    auto reservation = br->reserve_or_fail(nbytes, MemoryType::DEVICE);
    auto buffer = br->allocate(stream, std::move(reservation));

    RAPIDSMPF_EXPECTS(
        buffer->mem_type() == MemoryType::DEVICE,
        "make_packed_from_host expects DEVICE buffer"
    );
    RAPIDSMPF_EXPECTS(
        buffer->size == nbytes, "unexpected buffer size in make_packed_from_host"
    );

    buffer->write_access([&](std::byte* buf_data, rmm::cuda_stream_view s) {
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(buf_data, data, nbytes, cudaMemcpyHostToDevice, s.value())
        );
    });

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(std::uint64_t));
    auto* meta_ptr = reinterpret_cast<std::uint64_t*>(metadata->data());
    *meta_ptr = static_cast<std::uint64_t>(count);

    return PackedData{std::move(metadata), std::move(buffer)};
}

/**
 * @brief Helper to copy a device-backed `PackedData` into a host vector of T.
 */
template <typename T>
std::vector<T> unpack_to_host(PackedData& pd) {
    RAPIDSMPF_EXPECTS(pd.data, "unpack_to_host encountered null data buffer");
    auto* buf = pd.data.get();
    auto const nbytes = buf->size;
    RAPIDSMPF_EXPECTS(
        nbytes % sizeof(T) == 0,
        "unpack_to_host: buffer size is not a multiple of sizeof(T)"
    );

    auto const count = nbytes / sizeof(T);
    std::vector<T> out(count);

    RAPIDSMPF_EXPECTS(
        buf->mem_type() == MemoryType::DEVICE,
        "unpack_to_host expects DEVICE-backed buffer"
    );

    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        out.data(), buf->data(), nbytes, cudaMemcpyDeviceToHost, buf->stream().value()
    ));
    buf->stream().synchronize();

    return out;
}

}  // namespace

extern Environment* GlobalEnvironment;

/**
 * @brief Base fixture providing a shared communicator, progress thread and buffer
 * resource for AllReduce tests.
 */
class BaseAllReduceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        GlobalEnvironment->barrier();

        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = std::make_unique<rapidsmpf::BufferResource>(mr.get());
        comm = GlobalEnvironment->comm_.get();
    }

    void TearDown() override {
        br.reset();
        mr.reset();
        GlobalEnvironment->barrier();
    }

    rapidsmpf::Communicator* comm;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

// Test simple shutdown (construct and destruct without inserting).
TEST_F(BaseAllReduceTest, shutdown) {
    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{99},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
    );
}

TEST_F(BaseAllReduceTest, timeout) {
    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{98},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
    );

    EXPECT_THROW(
        std::ignore = allreduce.wait_and_extract(std::chrono::milliseconds{20}),
        std::runtime_error
    );

    allreduce.insert_finished();
    auto result = allreduce.wait_and_extract();
    EXPECT_EQ(result.size(), 0);
}

class AllReduceIntSumTest : public BaseAllReduceTest,
                            public ::testing::WithParamInterface<std::tuple<int, int>> {
  protected:
    void SetUp() override {
        BaseAllReduceTest::SetUp();
        std::tie(n_elements, n_inserts) = GetParam();
    }

    int n_elements{};
    int n_inserts{};
};

// Parameterized test for different element counts and numbers of inserts.
INSTANTIATE_TEST_SUITE_P(
    AllReduceIntSum,
    AllReduceIntSumTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(0, 1, 10)  // n_inserts
    ),
    [](const ::testing::TestParamInfo<AllReduceIntSumTest::ParamType>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "_n_inserts_"
               + std::to_string(std::get<1>(info.param));
    }
);

TEST_P(AllReduceIntSumTest, basic_allreduce_sum_int) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
    );

    for (int i = 0; i < n_inserts; i++) {
        std::vector<int> data(std::max(0, n_elements));
        for (int j = 0; j < n_elements; j++) {
            data[j] = i * 10 + this_rank;
        }
        auto packed = make_packed_from_host(br.get(), data.data(), data.size());
        allreduce.insert(static_cast<std::uint64_t>(i), std::move(packed));
    }

    allreduce.insert_finished();

    auto results = allreduce.wait_and_extract();
    if (n_inserts == 0) {
        EXPECT_TRUE(results.empty());
        EXPECT_TRUE(allreduce.finished());
        return;
    }

    ASSERT_EQ(static_cast<std::size_t>(n_inserts), results.size());

    for (int i = 0; i < n_inserts; i++) {
        auto& pd = results[static_cast<std::size_t>(i)];
        auto reduced = unpack_to_host<int>(pd);
        ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
        int const expected_value =
            static_cast<int>(nranks) * (i * 10) + (nranks * (nranks - 1)) / 2;
        for (int j = 0; j < n_elements; j++) {
            EXPECT_EQ(reduced[static_cast<std::size_t>(j)], expected_value);
        }
    }

    EXPECT_TRUE(allreduce.finished());
}

// Typed host tests covering multiple T / ReduceOp combinations.

template <typename Config>
class AllReduceTypedOpsTest : public BaseAllReduceTest {
  public:
    using value_type = typename Config::value_type;
    static constexpr ReduceOp op = Config::reduce_op_id;
};

using AllReduceConfigs = ::testing::Types<
    AllReduceConfig<int, ReduceOp::SUM>,
    AllReduceConfig<int, ReduceOp::PROD>,
    AllReduceConfig<int, ReduceOp::MIN>,
    AllReduceConfig<int, ReduceOp::MAX>,
    AllReduceConfig<float, ReduceOp::SUM>,
    AllReduceConfig<float, ReduceOp::PROD>,
    AllReduceConfig<float, ReduceOp::MIN>,
    AllReduceConfig<float, ReduceOp::MAX>,
    AllReduceConfig<double, ReduceOp::SUM>,
    AllReduceConfig<double, ReduceOp::PROD>,
    AllReduceConfig<double, ReduceOp::MIN>,
    AllReduceConfig<double, ReduceOp::MAX>,
    AllReduceConfig<std::uint64_t, ReduceOp::SUM>,
    AllReduceConfig<std::uint64_t, ReduceOp::PROD>,
    AllReduceConfig<std::uint64_t, ReduceOp::MIN>,
    AllReduceConfig<std::uint64_t, ReduceOp::MAX>>;

TYPED_TEST_SUITE(AllReduceTypedOpsTest, AllReduceConfigs);

TYPED_TEST(AllReduceTypedOpsTest, basic_host_allreduce) {
    using Config = TypeParam;
    using T = typename Config::value_type;
    constexpr auto op = Config::reduce_op_id;

    auto this_rank = this->comm->rank();
    auto nranks = this->comm->nranks();

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{1},
        this->br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<T, op>()
    );

    int constexpr n_elements{8};
    int constexpr n_inserts{3};

    for (int i = 0; i < n_inserts; i++) {
        std::vector<T> data(n_elements);
        for (int j = 0; j < n_elements; j++) {
            data[j] = make_input_value<T>(this_rank, i, j);
        }
        auto packed = make_packed_from_host<T>(this->br.get(), data.data(), data.size());
        allreduce.insert(static_cast<std::uint64_t>(i), std::move(packed));
    }

    allreduce.insert_finished();
    auto results = allreduce.wait_and_extract();

    ASSERT_EQ(static_cast<std::size_t>(n_inserts), results.size());

    for (int i = 0; i < n_inserts; i++) {
        auto& pd = results[static_cast<std::size_t>(i)];
        auto reduced = unpack_to_host<T>(pd);
        ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
        for (int j = 0; j < n_elements; j++) {
            T expected = make_input_value<T>(0, i, j);
            for (int r = 1; r < nranks; ++r) {
                expected = Combiner<op>::template apply<T>(
                    expected, make_input_value<T>(r, i, j)
                );
            }
            EXPECT_EQ(reduced[static_cast<std::size_t>(j)], expected);
        }
    }
}

/**
 * @brief Test demonstrating AllReduce over a user-defined type with a custom
 * reduction operator.
 */
class AllReduceCustomTypeTest : public BaseAllReduceTest {};

TEST_F(AllReduceCustomTypeTest, custom_struct_allreduce) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    // Use a custom device-side reduction kernel for CustomValue.
    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{5},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        make_custom_value_reduce_kernel()
    );

    int constexpr n_elements{4};
    int constexpr n_inserts{3};

    // Each rank contributes the same logical layout of CustomValue, staged into
    // device-backed PackedData.
    for (int i = 0; i < n_inserts; ++i) {
        std::vector<CustomValue> data(n_elements);
        for (int j = 0; j < n_elements; ++j) {
            data[static_cast<std::size_t>(j)].value =
                make_input_value<int>(this_rank, i, j);
            data[static_cast<std::size_t>(j)].weight = this_rank * 10 + j;
        }

        auto packed =
            make_packed_from_host<CustomValue>(br.get(), data.data(), data.size());

        allreduce.insert(static_cast<std::uint64_t>(i), std::move(packed));
    }

    allreduce.insert_finished();
    auto results = allreduce.wait_and_extract();

    ASSERT_EQ(static_cast<std::size_t>(n_inserts), results.size());

    for (int i = 0; i < n_inserts; ++i) {
        auto& pd = results[static_cast<std::size_t>(i)];
        auto reduced = unpack_to_host<CustomValue>(pd);
        ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());

        for (int j = 0; j < n_elements; ++j) {
            // Expected value is SUM over ranks of make_input_value<int>(rank, i, j).
            int expected_value = make_input_value<int>(0, i, j);
            for (int r = 1; r < nranks; ++r) {
                expected_value += make_input_value<int>(r, i, j);
            }

            // Expected weight is MIN over ranks of (rank * 10 + j).
            int expected_weight = 0 * 10 + j;
            for (int r = 1; r < nranks; ++r) {
                expected_weight = std::min(expected_weight, r * 10 + j);
            }

            auto const& cv = reduced[static_cast<std::size_t>(j)];
            EXPECT_EQ(cv.value, expected_value);
            EXPECT_EQ(cv.weight, expected_weight);
        }
    }
}

class AllReduceNonUniformInsertsTest : public BaseAllReduceTest {};

TEST_F(AllReduceNonUniformInsertsTest, non_uniform_inserts) {
    auto this_rank = comm->rank();
    constexpr int n_elements = 5;

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{6},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
    );

    // Rank r inserts r messages -> non-uniform across ranks.
    auto n_inserts = this_rank;
    for (int i = 0; i < n_inserts; ++i) {
        std::vector<int> data(n_elements);
        for (int j = 0; j < n_elements; ++j) {
            data[static_cast<std::size_t>(j)] = make_input_value<int>(this_rank, i, j);
        }
        auto packed = make_packed_from_host<int>(br.get(), data.data(), data.size());
        allreduce.insert(static_cast<std::uint64_t>(i), std::move(packed));
    }

    allreduce.insert_finished();

    // AllReduce currently requires each rank to contribute the same number of
    // messages. With non-uniform inserts this invariant is violated and the
    // reduction is expected to throw when we attempt to extract results.
    EXPECT_THROW(std::ignore = allreduce.wait_and_extract(), std::runtime_error);
}
