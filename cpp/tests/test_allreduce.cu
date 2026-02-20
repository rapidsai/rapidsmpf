/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/functional.h>

#include <cuda/std/functional>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

#include "environment.hpp"

using rapidsmpf::MemoryType;
using rapidsmpf::OpID;
using rapidsmpf::coll::AllReduce;
using rapidsmpf::coll::ReduceOperator;

namespace {

template <typename T>
using SumOp = cuda::std::plus<T>;

template <typename T>
using ProdOp = cuda::std::multiplies<T>;

template <typename T>
using MinOp = cuda::minimum<T>;

template <typename T>
using MaxOp = cuda::maximum<T>;

template <typename Op>
struct OpName;

template <typename T>
struct OpName<cuda::std::plus<T>> {
    static constexpr const char* value() {
        return "sum";
    }
};

template <typename T>
struct OpName<cuda::std::multiplies<T>> {
    static constexpr const char* value() {
        return "prod";
    }
};

template <typename T>
struct OpName<cuda::minimum<T>> {
    static constexpr const char* value() {
        return "min";
    }
};

template <typename T>
struct OpName<cuda::maximum<T>> {
    static constexpr const char* value() {
        return "max";
    }
};

template <typename Op>
constexpr const char* ToStringOp() {
    return OpName<Op>::value();
}

constexpr const char* ToString(MemoryType mem_type) {
    switch (mem_type) {
    case MemoryType::HOST:
        return "host";
    case MemoryType::DEVICE:
        return "device";
    default:
        return "unknown_mem_type";
    }
}

template <typename T>
T make_input_value(int rank, int elem_idx) {
    // Simple positive pattern valid for all tested arithmetic types.
    auto base = static_cast<std::uint64_t>((rank + 1) * 100 + (elem_idx + 1) * 10);
    return static_cast<T>(base + elem_idx);
}

/**
 * @brief Simple user-defined type used to demonstrate custom allreduce operators.
 *
 * The reduction we implement for this type is:
 *  - `value` field: summed across ranks
 *  - `weight` field: minimum across ranks
 */
struct CustomValue {
    int value{};
    int weight{};
};

/**
 * @brief Factory for a host-side reduction operator for `CustomValue`.
 *
 * The returned operator expects the buffer to contain a contiguous array
 * of `CustomValue` in host memory, with equal sizes for all ranks.
 */
ReduceOperator make_custom_value_reduce_operator_host() {
    return rapidsmpf::coll::detail::make_host_reduce_operator<CustomValue>(
        [](CustomValue a, CustomValue b) -> CustomValue {
            return {
                .value = a.value + b.value, .weight = cuda::std::min(a.weight, b.weight)
            };
        }
    );
}

ReduceOperator make_custom_value_reduce_operator_device() {
    return rapidsmpf::coll::detail::make_device_reduce_operator<CustomValue>(
        [] __device__(CustomValue a, CustomValue b) -> CustomValue {
            return {
                .value = a.value + b.value, .weight = cuda::std::min(a.weight, b.weight)
            };
        }
    );
}

/**
 * @brief Helper to create a buffer from host memory, backed by either host or
 * device memory.
 */
template <typename T>
std::unique_ptr<rapidsmpf::Buffer> make_buffer(
    rapidsmpf::BufferResource* br, T const* data, std::size_t count, MemoryType mem_type
) {
    auto const nbytes = count * sizeof(T);
    auto stream = br->stream_pool().get_stream();
    auto reservation = br->reserve_or_fail(nbytes, mem_type);
    auto buffer = br->allocate(stream, std::move(reservation));

    RAPIDSMPF_EXPECTS(
        buffer->mem_type() == mem_type, "make_buffer: buffer memory type mismatch"
    );
    RAPIDSMPF_EXPECTS(buffer->size == nbytes, "unexpected buffer size in make_buffer");

    auto* raw_ptr = buffer->exclusive_data_access();
    RAPIDSMPF_CUDA_TRY(
        cudaMemcpyAsync(raw_ptr, data, nbytes, cudaMemcpyDefault, stream.value())
    );
    stream.synchronize();
    buffer->unlock();

    return buffer;
}

/**
 * @brief Helper to copy a buffer (host or device-backed) into a host vector of T.
 */
template <typename T>
std::vector<T> unpack_to_host(rapidsmpf::Buffer& buffer) {
    auto* buf = &buffer;
    auto const nbytes = buf->size;
    RAPIDSMPF_EXPECTS(
        nbytes % sizeof(T) == 0,
        "unpack_to_host: buffer size is not a multiple of sizeof(T)"
    );

    auto const count = nbytes / sizeof(T);
    std::vector<T> out(count);
    // Buffer is only guaranteed to be stream ordered on stream, not fully ready, so sync
    // first.
    buffer.stream().synchronize();
    auto* raw_ptr = buf->exclusive_data_access();
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        out.data(), raw_ptr, nbytes, cudaMemcpyDefault, buf->stream().value()
    ));
    buf->stream().synchronize();
    buf->unlock();

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

TEST_F(BaseAllReduceTest, timeout_interleaved) {
    auto const rank = comm->rank();
    auto const size = comm->nranks();
    std::vector<int> data1(1, rank + 1);
    auto in_buffer1 =
        make_buffer<int>(br.get(), data1.data(), data1.size(), MemoryType::HOST);
    auto reservation = br->reserve_or_fail(in_buffer1->size, in_buffer1->mem_type());
    auto out_buffer1 = br->allocate(in_buffer1->size, in_buffer1->stream(), reservation);

    std::vector<float> data2(1, rank + 7);
    auto in_buffer2 =
        make_buffer<float>(br.get(), data2.data(), data2.size(), MemoryType::DEVICE);
    reservation = br->reserve_or_fail(in_buffer2->size, in_buffer2->mem_type());
    auto out_buffer2 = br->allocate(in_buffer2->size, in_buffer2->stream(), reservation);

    if (rank % 2 == 0) {
        AllReduce allreduce1(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            std::move(in_buffer1),
            std::move(out_buffer1),
            OpID{0},
            rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{})
        );
        if (size > 1) {
            // Odd ranks have not yet started the matching allreduce because they're doing
            // the "second" one first.
            EXPECT_THROW(
                std::ignore = allreduce1.wait_and_extract(std::chrono::milliseconds{20}),
                std::runtime_error
            );
        }
        AllReduce allreduce2(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            std::move(in_buffer2),
            std::move(out_buffer2),
            OpID{1},
            rapidsmpf::coll::detail::make_device_reduce_operator<float>(SumOp<float>{})
        );
        // OK, now we can get the result from the "second" allreduce
        std::tie(in_buffer2, out_buffer2) = allreduce2.wait_and_extract();
        // And now the odd ranks are release so they kick off the "first" reduce and this
        // should succeed.
        std::tie(in_buffer1, out_buffer1) = allreduce1.wait_and_extract();
    } else {
        AllReduce allreduce2(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            std::move(in_buffer2),
            std::move(out_buffer2),
            OpID{1},
            rapidsmpf::coll::detail::make_device_reduce_operator<float>(SumOp<float>{})
        );
        std::tie(in_buffer2, out_buffer2) = allreduce2.wait_and_extract();
        AllReduce allreduce1(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            std::move(in_buffer1),
            std::move(out_buffer1),
            OpID{0},
            rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{})
        );
        std::tie(in_buffer1, out_buffer1) = allreduce1.wait_and_extract();
    }
    auto result1 = unpack_to_host<int>(*out_buffer1);
    auto result2 = unpack_to_host<float>(*out_buffer2);
    EXPECT_EQ(result1.size(), 1);
    EXPECT_EQ(result2.size(), 1);
    EXPECT_EQ(result1[0], size * (size + 1) / 2);
    EXPECT_FLOAT_EQ(result2[0], static_cast<float>(6 * size + (size * (size + 1)) / 2));
}

class AllReduceIntSumTest
    : public BaseAllReduceTest,
      public ::testing::WithParamInterface<std::tuple<int, MemoryType>> {
  protected:
    void SetUp() override {
        BaseAllReduceTest::SetUp();
        n_elements = std::get<0>(GetParam());
        mem_type = std::get<1>(GetParam());
    }

    int n_elements{};
    MemoryType mem_type{};
};

INSTANTIATE_TEST_SUITE_P(
    AllReduceIntSum,
    AllReduceIntSumTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(
            MemoryType::HOST
#ifdef __CUDACC__
            ,
            MemoryType::DEVICE
#endif
        )
    ),
    [](const ::testing::TestParamInfo<std::tuple<int, MemoryType>>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "_"
               + ToString(std::get<1>(info.param));
    }
);

TEST_P(AllReduceIntSumTest, basic_allreduce_sum_int) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    // Choose operator based on reduction type
    ReduceOperator kernel =
        mem_type == MemoryType::DEVICE
            ? rapidsmpf::coll::detail::make_device_reduce_operator<int>(SumOp<int>{})
            : rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{});

    std::vector<int> data(std::max(0, n_elements));
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }

    auto in_buffer = make_buffer<int>(br.get(), data.data(), data.size(), mem_type);
    auto reservation = br->reserve_or_fail(in_buffer->size, in_buffer->mem_type());
    auto out_buffer = br->allocate(in_buffer->size, in_buffer->stream(), reservation);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        std::move(in_buffer),
        std::move(out_buffer),
        OpID{0},
        std::move(kernel)
    );

    auto [in_result, out_result] = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<int>(*out_result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    // Expected value is sum of all ranks (0 + 1 + 2 + ... + nranks-1)
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());
}

template <typename T, typename Op, MemoryType MemType>
struct AllReduceCase {
    using value_type = T;
    using op_type = Op;
    static constexpr MemoryType mem_type = MemType;
};

#define HOST_BUFFER_REDUCTION_CASES(T, OP) AllReduceCase<T, OP, MemoryType::HOST>

#ifdef __CUDACC__
#define DEVICE_BUFFER_REDUCTION_CASES(T, OP) AllReduceCase<T, OP, MemoryType::DEVICE>
#else
#define DEVICE_BUFFER_REDUCTION_CASES(T, OP)
#endif

#define ALL_BUFFER_REDUCTION_CASES(T, OP) \
    HOST_BUFFER_REDUCTION_CASES(T, OP), DEVICE_BUFFER_REDUCTION_CASES(T, OP)

using AllReduceCases = ::testing::Types<
    ALL_BUFFER_REDUCTION_CASES(int, SumOp<int>),
    ALL_BUFFER_REDUCTION_CASES(int, ProdOp<int>),
    ALL_BUFFER_REDUCTION_CASES(int, MinOp<int>),
    ALL_BUFFER_REDUCTION_CASES(int, MaxOp<int>),
    ALL_BUFFER_REDUCTION_CASES(float, SumOp<float>),
    ALL_BUFFER_REDUCTION_CASES(float, ProdOp<float>),
    ALL_BUFFER_REDUCTION_CASES(float, MinOp<float>),
    ALL_BUFFER_REDUCTION_CASES(float, MaxOp<float>),
    ALL_BUFFER_REDUCTION_CASES(double, SumOp<double>),
    ALL_BUFFER_REDUCTION_CASES(double, ProdOp<double>),
    ALL_BUFFER_REDUCTION_CASES(double, MinOp<double>),
    ALL_BUFFER_REDUCTION_CASES(double, MaxOp<double>),
    ALL_BUFFER_REDUCTION_CASES(std::uint64_t, SumOp<std::uint64_t>),
    ALL_BUFFER_REDUCTION_CASES(std::uint64_t, ProdOp<std::uint64_t>),
    ALL_BUFFER_REDUCTION_CASES(std::uint64_t, MinOp<std::uint64_t>),
    ALL_BUFFER_REDUCTION_CASES(std::uint64_t, MaxOp<std::uint64_t>)>;

#undef ALL_BUFFER_REDUCTION_CASES
#undef HOST_BUFFER_REDUCTION_CASES
#undef DEVICE_BUFFER_REDUCTION_CASES

template <typename Case>
class AllReduceTypedOpsTest : public BaseAllReduceTest {
  public:
    using value_type = typename Case::value_type;
    using op_type = typename Case::op_type;
    static constexpr auto mem_type = Case::mem_type;
};

struct AllReduceTypedOpsTestName {
    template <typename Case>
    static std::string GetName(int) {
        std::string type_name =
            ::testing::internal::GetTypeName<typename Case::value_type>();
        return type_name + "_" + ToStringOp<typename Case::op_type>() + "_"
               + ToString(Case::mem_type);
    }
};

TYPED_TEST_SUITE(AllReduceTypedOpsTest, AllReduceCases, AllReduceTypedOpsTestName);

TYPED_TEST(AllReduceTypedOpsTest, basic_allreduce) {
    using Case = TypeParam;
    using T = typename Case::value_type;
    using Op = typename Case::op_type;

    auto this_rank = this->comm->rank();
    auto nranks = this->comm->nranks();

    // Convert op to device version if needed
    ReduceOperator kernel = [&]() -> ReduceOperator {
        if constexpr (Case::mem_type == MemoryType::DEVICE) {
            return rapidsmpf::coll::detail::make_device_reduce_operator<T>(Op{});
        }

        return rapidsmpf::coll::detail::make_host_reduce_operator<T>(Op{});
    }();

    int constexpr n_elements{8};

    std::vector<T> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = make_input_value<T>(this_rank, j);
    }

    MemoryType mem_type = Case::mem_type;

    auto in_buffer = make_buffer<T>(this->br.get(), data.data(), data.size(), mem_type);
    auto reservation = this->br->reserve_or_fail(in_buffer->size, in_buffer->mem_type());
    auto out_buffer =
        this->br->allocate(in_buffer->size, in_buffer->stream(), reservation);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        std::move(in_buffer),
        std::move(out_buffer),
        OpID{0},
        std::move(kernel)
    );
    auto [in_result, out_result] = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<T>(*out_result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    std::vector<T> expected;
    expected.reserve(n_elements);
    Op const op{};
    for (int j = 0; j < n_elements; j++) {
        T value = make_input_value<T>(0, j);
        for (int r = 1; r < nranks; ++r) {
            value = op(value, make_input_value<T>(r, j));
        }
        expected.push_back(value);
    }

    if constexpr (std::is_floating_point_v<T>) {
        auto near_matcher = ::testing::Truly([](auto const& pair) {
            auto const& actual = std::get<0>(pair);
            auto const& exp = std::get<1>(pair);
            auto const tol = std::abs(exp) * 1e-5 + 1e-5;
            return std::abs(actual - exp) <= tol;
        });
        EXPECT_THAT(reduced, ::testing::Pointwise(near_matcher, expected));
    } else {
        EXPECT_THAT(reduced, ::testing::ElementsAreArray(expected));
    }
}

/**
 * @brief Test demonstrating AllReduce over a user-defined type with a custom
 * reduction operator.
 */
class AllReduceCustomTypeTest : public BaseAllReduceTest,
                                public ::testing::WithParamInterface<MemoryType> {};

TEST_P(AllReduceCustomTypeTest, custom_struct_allreduce) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    int constexpr n_elements{4};

    auto const& mem_type = GetParam();

    ReduceOperator kernel = mem_type == MemoryType::DEVICE
                                ? make_custom_value_reduce_operator_device()
                                : make_custom_value_reduce_operator_host();

    // Each rank contributes the same logical layout of CustomValue
    std::vector<CustomValue> data(n_elements);
    for (int j = 0; j < n_elements; ++j) {
        data[static_cast<std::size_t>(j)].value = make_input_value<int>(this_rank, j);
        data[static_cast<std::size_t>(j)].weight = this_rank * 10 + j;
    }

    auto in_buffer =
        make_buffer<CustomValue>(br.get(), data.data(), data.size(), mem_type);
    auto reservation = br->reserve_or_fail(in_buffer->size, in_buffer->mem_type());
    auto out_buffer = br->allocate(in_buffer->size, in_buffer->stream(), reservation);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        std::move(in_buffer),
        std::move(out_buffer),
        OpID{0},
        std::move(kernel)
    );
    auto [in_result, out_result] = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<CustomValue>(*out_result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());

    std::vector<CustomValue> expected;
    expected.reserve(n_elements);
    for (int j = 0; j < n_elements; ++j) {
        // Expected value is SUM over ranks of make_input_value<int>(rank, j).
        int expected_value = make_input_value<int>(0, j);
        for (int r = 1; r < nranks; ++r) {
            expected_value += make_input_value<int>(r, j);
        }

        // Expected weight is MIN over ranks of (rank * 10 + j).
        int expected_weight = 0 * 10 + j;
        for (int r = 1; r < nranks; ++r) {
            expected_weight = std::min(expected_weight, r * 10 + j);
        }

        expected.emplace_back(expected_value, expected_weight);
    }

    auto eq_custom_value = ::testing::Truly([](auto const& pair) {
        auto const& actual = std::get<0>(pair);
        auto const& exp = std::get<1>(pair);
        return actual.value == exp.value && actual.weight == exp.weight;
    });
    EXPECT_THAT(reduced, ::testing::Pointwise(eq_custom_value, expected));
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceCustomTypeConfigs,
    AllReduceCustomTypeTest,
    ::testing::Values(MemoryType::HOST, MemoryType::DEVICE),
    [](const ::testing::TestParamInfo<MemoryType>& info) { return ToString(info.param); }
);

class AllReduceFinishedCallbackTest : public BaseAllReduceTest {};

TEST_F(AllReduceFinishedCallbackTest, finished_callback_invoked) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();
    constexpr int n_elements = 10;

    std::atomic<bool> callback_called{false};
    std::atomic<int> callback_count{0};

    std::vector<int> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }
    auto in_buffer = make_buffer(br.get(), data.data(), data.size(), MemoryType::HOST);
    auto reservation = br->reserve_or_fail(in_buffer->size, in_buffer->mem_type());
    auto out_buffer = br->allocate(in_buffer->size, in_buffer->stream(), reservation);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        std::move(in_buffer),
        std::move(out_buffer),
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        [&callback_called, &callback_count]() {
            callback_called.store(true, std::memory_order_release);
            callback_count.fetch_add(1, std::memory_order_relaxed);
        }
    );

    auto [in_result, out_result] = allreduce.wait_and_extract();

    // After wait_and_extract completes, callback should have been called exactly once
    EXPECT_TRUE(callback_called.load(std::memory_order_acquire));
    EXPECT_EQ(callback_count.load(std::memory_order_acquire), 1);

    auto reduced = unpack_to_host<int>(*out_result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());
}

TEST_F(AllReduceFinishedCallbackTest, wait_and_extract_multiple_times) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();
    constexpr int n_elements = 5;

    std::vector<int> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }
    auto in_buffer = make_buffer(br.get(), data.data(), data.size(), MemoryType::DEVICE);
    auto reservation = br->reserve_or_fail(in_buffer->size, in_buffer->mem_type());
    auto out_buffer = br->allocate(in_buffer->size, in_buffer->stream(), reservation);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        std::move(in_buffer),
        std::move(out_buffer),
        OpID{0},
        rapidsmpf::coll::detail::make_device_reduce_operator<int>(SumOp<int>{})
    );

    // First call should succeed
    auto [in_result1, out_result1] = allreduce.wait_and_extract();
    auto reduced1 = unpack_to_host<int>(*out_result1);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced1.size());
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced1, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());

    // Second call: data already extracted, should throw
    EXPECT_THROW(std::ignore = allreduce.wait_and_extract(), std::runtime_error);
}
