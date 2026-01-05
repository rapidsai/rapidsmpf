/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/functional.h>

#include <cuda/std/functional>

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
using rapidsmpf::coll::ReduceOperator;
using rapidsmpf::coll::ReduceOperatorType;

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

template <typename Op, typename T>
T apply_op(Op op, T a, T b) {
    return op(a, b);
}

template <typename T>
T make_input_value(int rank, int elem_idx) {
    // Simple positive pattern valid for all tested arithmetic types.
    auto base = static_cast<std::uint64_t>((rank + 1) * 100 + (elem_idx + 1) * 10);
    return static_cast<T>(base + elem_idx);
}

/**
 * @brief Factory for a host-side reduction operator for `CustomValue`.
 *
 * The returned operator expects `PackedData::data` to contain a contiguous array
 * of `CustomValue` in host memory, with equal sizes for all ranks.
 */
ReduceOperator make_custom_value_reduce_operator_host() {
    struct CustomValueOp {
        CustomValue operator()(CustomValue const& a, CustomValue const& b) const {
            CustomValue result;
            result.value = a.value + b.value;
            result.weight = std::min(a.weight, b.weight);
            return result;
        }
    };

    return rapidsmpf::coll::detail::make_host_reduce_operator<CustomValue>(
        CustomValueOp{}
    );
}

/**
 * @brief Helper to create a `PackedData` from host memory, backed by either host or
 * device memory.
 */
template <typename T>
PackedData make_packed(
    rapidsmpf::BufferResource* br, T const* data, std::size_t count, MemoryType mem_type
) {
    auto const nbytes = count * sizeof(T);
    auto stream = br->stream_pool().get_stream();
    auto reservation = br->reserve_or_fail(nbytes, mem_type);
    auto buffer = br->allocate(stream, std::move(reservation));

    RAPIDSMPF_EXPECTS(
        buffer->mem_type() == mem_type, "make_packed: buffer memory type mismatch"
    );
    RAPIDSMPF_EXPECTS(buffer->size == nbytes, "unexpected buffer size in make_packed");

    if (mem_type == MemoryType::DEVICE) {
        buffer->write_access([&](std::byte* buf_data, rmm::cuda_stream_view s) {
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(buf_data, data, nbytes, cudaMemcpyHostToDevice, s.value())
            );
        });
    } else {
        auto* raw_ptr = buffer->exclusive_data_access();
        std::memcpy(raw_ptr, data, nbytes);
        buffer->unlock();
    }

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(std::uint64_t));
    auto* meta_ptr = reinterpret_cast<std::uint64_t*>(metadata->data());
    *meta_ptr = static_cast<std::uint64_t>(count);

    return PackedData{std::move(metadata), std::move(buffer)};
}

/**
 * @brief Helper to copy a `PackedData` (host or device-backed) into a host vector of T.
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

    if (buf->mem_type() == MemoryType::DEVICE) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            out.data(), buf->data(), nbytes, cudaMemcpyDeviceToHost, buf->stream().value()
        ));
        buf->stream().synchronize();
    } else if (buf->mem_type() == MemoryType::HOST) {
        auto* raw_ptr = buf->exclusive_data_access();
        std::memcpy(out.data(), raw_ptr, nbytes);
        buf->unlock();
    } else {
        RAPIDSMPF_FAIL("unpack_to_host: unsupported memory type");
    }

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

TEST_F(BaseAllReduceTest, shutdown) {
    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );
}

TEST_F(BaseAllReduceTest, timeout) {
    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );

    std::vector<int> data(1, 42);
    auto packed = make_packed(br.get(), data.data(), data.size(), MemoryType::DEVICE);
    allreduce.insert(std::move(packed));

    auto result = allreduce.wait_and_extract();
}

struct MemoryReductionConfig {
    enum BufferType {
        ALL_HOST,
        ALL_DEVICE,
        MIXED
    };

    enum ReductionType {
        HOST_REDUCTION,
        DEVICE_REDUCTION
    };

    BufferType buffer_type;
    ReductionType reduction_type;

    std::string ToString() const {
        std::string buf_str = (buffer_type == ALL_HOST)     ? "all_host"
                              : (buffer_type == ALL_DEVICE) ? "all_device"
                                                            : "mixed";
        std::string red_str =
            (reduction_type == HOST_REDUCTION) ? "host_reduction" : "device_reduction";
        return buf_str + "_" + red_str;
    }
};

class AllReduceIntSumTest
    : public BaseAllReduceTest,
      public ::testing::WithParamInterface<std::tuple<int, MemoryReductionConfig>> {
  protected:
    void SetUp() override {
        BaseAllReduceTest::SetUp();
        n_elements = std::get<0>(GetParam());
        config = std::get<1>(GetParam());
    }

    int n_elements{};
    MemoryReductionConfig config{};
};

INSTANTIATE_TEST_SUITE_P(
    AllReduceIntSum,
    AllReduceIntSumTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_HOST, MemoryReductionConfig::HOST_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::HOST_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::MIXED, MemoryReductionConfig::HOST_REDUCTION
            }
#ifdef __CUDACC__
            ,
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::DEVICE_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_HOST, MemoryReductionConfig::DEVICE_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::MIXED, MemoryReductionConfig::DEVICE_REDUCTION
            }
#endif
        )
    ),
    [](const ::testing::TestParamInfo<std::tuple<int, MemoryReductionConfig>>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "_"
               + std::get<1>(info.param).ToString();
    }
);

TEST_P(AllReduceIntSumTest, basic_allreduce_sum_int) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    // Choose operator based on reduction type
    ReduceOperator kernel =
        (config.reduction_type == MemoryReductionConfig::DEVICE_REDUCTION)
            ? rapidsmpf::coll::detail::make_device_reduce_operator<int>(SumOp<int>{})
            : rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{});

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        std::move(kernel),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );

    std::vector<int> data(std::max(0, n_elements));
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }

    // Choose buffer type based on config
    PackedData packed =
        (config.buffer_type == MemoryReductionConfig::ALL_HOST)
            ? make_packed<int>(br.get(), data.data(), data.size(), MemoryType::HOST)
        : (config.buffer_type == MemoryReductionConfig::ALL_DEVICE)
            ? make_packed<int>(br.get(), data.data(), data.size(), MemoryType::DEVICE)
        : (this_rank % 2 == 0)  // MIXED: even ranks use host, odd ranks use device
            ? make_packed<int>(br.get(), data.data(), data.size(), MemoryType::HOST)
            : make_packed<int>(br.get(), data.data(), data.size(), MemoryType::DEVICE);

    allreduce.insert(std::move(packed));

    auto result = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<int>(result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    // Expected value is sum of all ranks (0 + 1 + 2 + ... + nranks-1)
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());
}

template <
    typename T,
    typename Op,
    MemoryReductionConfig::BufferType BufferType,
    MemoryReductionConfig::ReductionType ReductionType>
struct AllReduceCase {
    using value_type = T;
    using op_type = Op;
    static constexpr MemoryReductionConfig::BufferType buffer_type = BufferType;
    static constexpr MemoryReductionConfig::ReductionType reduction_type = ReductionType;
};

template <typename Op>
constexpr const char* ToStringOp() {
    return OpName<Op>::value();
}

constexpr const char* ToString(MemoryReductionConfig::BufferType bt) {
    switch (bt) {
    case MemoryReductionConfig::ALL_HOST:
        return "all_host";
    case MemoryReductionConfig::ALL_DEVICE:
        return "all_device";
    case MemoryReductionConfig::MIXED:
        return "mixed";
    }
    return "unknown_buffer";
}

constexpr const char* ToString(MemoryReductionConfig::ReductionType rt) {
    switch (rt) {
    case MemoryReductionConfig::HOST_REDUCTION:
        return "host_red";
    case MemoryReductionConfig::DEVICE_REDUCTION:
        return "device_red";
    }
    return "unknown_reduction";
}

#define HOST_BUFFER_REDUCTION_CASES(T, OP)          \
    AllReduceCase<                                  \
        T,                                          \
        OP,                                         \
        MemoryReductionConfig::ALL_HOST,            \
        MemoryReductionConfig::HOST_REDUCTION>,     \
        AllReduceCase<                              \
            T,                                      \
            OP,                                     \
            MemoryReductionConfig::ALL_DEVICE,      \
            MemoryReductionConfig::HOST_REDUCTION>, \
        AllReduceCase<                              \
            T,                                      \
            OP,                                     \
            MemoryReductionConfig::MIXED,           \
            MemoryReductionConfig::HOST_REDUCTION>

#ifdef __CUDACC__
#define DEVICE_BUFFER_REDUCTION_CASES(T, OP)          \
    AllReduceCase<                                    \
        T,                                            \
        OP,                                           \
        MemoryReductionConfig::ALL_DEVICE,            \
        MemoryReductionConfig::DEVICE_REDUCTION>,     \
        AllReduceCase<                                \
            T,                                        \
            OP,                                       \
            MemoryReductionConfig::ALL_HOST,          \
            MemoryReductionConfig::DEVICE_REDUCTION>, \
        AllReduceCase<                                \
            T,                                        \
            OP,                                       \
            MemoryReductionConfig::MIXED,             \
            MemoryReductionConfig::DEVICE_REDUCTION>
#else
#define DEVICE_BUFFER_REDUCTION_CASES(T, OP)
#endif

#ifdef __CUDACC__
#define ALL_BUFFER_REDUCTION_CASES(T, OP) \
    HOST_BUFFER_REDUCTION_CASES(T, OP), DEVICE_BUFFER_REDUCTION_CASES(T, OP)
#else
#define ALL_BUFFER_REDUCTION_CASES(T, OP) HOST_BUFFER_REDUCTION_CASES(T, OP)
#endif

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
    static constexpr auto buffer_type = Case::buffer_type;
    static constexpr auto reduction_type = Case::reduction_type;
};

struct AllReduceTypedOpsTestName {
    template <typename Case>
    static std::string GetName(int) {
        std::string type_name =
            ::testing::internal::GetTypeName<typename Case::value_type>();
        return type_name + "_" + ToStringOp<typename Case::op_type>() + "_"
               + ToString(Case::buffer_type) + "_" + ToString(Case::reduction_type);
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
        if constexpr (Case::reduction_type == MemoryReductionConfig::DEVICE_REDUCTION) {
            return rapidsmpf::coll::detail::make_device_reduce_operator<T>(Op{});
        }

        return rapidsmpf::coll::detail::make_host_reduce_operator<T>(Op{});
    }();

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        std::move(kernel),
        this->br.get(),
        rapidsmpf::Statistics::disabled()
    );

    int constexpr n_elements{8};

    std::vector<T> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = make_input_value<T>(this_rank, j);
    }

    PackedData packed =
        (Case::buffer_type == MemoryReductionConfig::ALL_HOST)
            ? make_packed<T>(this->br.get(), data.data(), data.size(), MemoryType::HOST)
        : (Case::buffer_type == MemoryReductionConfig::ALL_DEVICE)
            ? make_packed<T>(this->br.get(), data.data(), data.size(), MemoryType::DEVICE)
        : (this_rank % 2 == 0)  // MIXED: even ranks use host, odd ranks use device
            ? make_packed<T>(this->br.get(), data.data(), data.size(), MemoryType::HOST)
            : make_packed<T>(
                  this->br.get(), data.data(), data.size(), MemoryType::DEVICE
              );

    allreduce.insert(std::move(packed));
    auto result = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<T>(result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    std::vector<T> expected;
    expected.reserve(n_elements);
    Op const op{};
    for (int j = 0; j < n_elements; j++) {
        T value = make_input_value<T>(0, j);
        for (int r = 1; r < nranks; ++r) {
            value = apply_op(op, value, make_input_value<T>(r, j));
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
struct CustomTypeTestConfig {
    bool is_device;
    MemoryType buffer_type;
    const char* name;
};

class AllReduceCustomTypeTest
    : public BaseAllReduceTest,
      public ::testing::WithParamInterface<CustomTypeTestConfig> {};

TEST_P(AllReduceCustomTypeTest, custom_struct_allreduce) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();

    int constexpr n_elements{4};

    auto const& config = GetParam();

    ReduceOperator kernel = config.is_device ? make_custom_value_reduce_operator_device()
                                             : make_custom_value_reduce_operator_host();

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        std::move(kernel),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );

    // Each rank contributes the same logical layout of CustomValue
    std::vector<CustomValue> data(n_elements);
    for (int j = 0; j < n_elements; ++j) {
        data[static_cast<std::size_t>(j)].value = make_input_value<int>(this_rank, j);
        data[static_cast<std::size_t>(j)].weight = this_rank * 10 + j;
    }

    auto packed =
        make_packed<CustomValue>(br.get(), data.data(), data.size(), config.buffer_type);

    allreduce.insert(std::move(packed));
    auto result = allreduce.wait_and_extract();

    auto reduced = unpack_to_host<CustomValue>(result);
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

        expected.push_back(CustomValue{expected_value, expected_weight});
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
    ::testing::Values(
        CustomTypeTestConfig{false, MemoryType::HOST, "host_reduction_with_host_buffers"},
        CustomTypeTestConfig{
            true, MemoryType::DEVICE, "device_reduction_with_device_buffers"
        }
    ),
    [](const ::testing::TestParamInfo<CustomTypeTestConfig>& info) {
        return std::string(info.param.name);
    }
);

class AllReduceNonUniformInsertsTest : public BaseAllReduceTest {};

TEST_F(AllReduceNonUniformInsertsTest, non_uniform_inserts) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    auto this_rank = comm->rank();
    constexpr int n_elements = 5;

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );

    // Test that some ranks not inserting causes failure.
    // Only rank 0 inserts data, other ranks don't.
    if (this_rank == 0) {
        std::vector<int> data(n_elements);
        for (int j = 0; j < n_elements; ++j) {
            data[static_cast<std::size_t>(j)] = make_input_value<int>(this_rank, j);
        }
        auto packed =
            make_packed<int>(br.get(), data.data(), data.size(), MemoryType::DEVICE);
        allreduce.insert(std::move(packed));
    }

    // Should fail since not all ranks contributed
    EXPECT_THROW(
        std::ignore = allreduce.wait_and_extract(std::chrono::milliseconds{20}),
        std::runtime_error
    );
}

class AllReduceFinishedCallbackTest : public BaseAllReduceTest {};

TEST_F(AllReduceFinishedCallbackTest, finished_callback_invoked) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();
    constexpr int n_elements = 10;

    std::atomic<bool> callback_called{false};
    std::atomic<int> callback_count{0};

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled(),
        [&callback_called, &callback_count]() {
            callback_called.store(true, std::memory_order_release);
            callback_count.fetch_add(1, std::memory_order_relaxed);
        }
    );

    // Initially callback should not be called
    EXPECT_FALSE(callback_called.load(std::memory_order_acquire));
    EXPECT_EQ(callback_count.load(std::memory_order_acquire), 0);

    std::vector<int> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }
    auto packed = make_packed(br.get(), data.data(), data.size(), MemoryType::DEVICE);
    allreduce.insert(std::move(packed));

    auto result = allreduce.wait_and_extract();

    // After wait_and_extract completes, callback should have been called exactly once
    EXPECT_TRUE(callback_called.load(std::memory_order_acquire));
    EXPECT_EQ(callback_count.load(std::memory_order_acquire), 1);

    auto reduced = unpack_to_host<int>(result);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());
}

TEST_F(AllReduceFinishedCallbackTest, finished_callback_not_called_without_insert) {
    std::atomic<bool> callback_called{false};

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled(),
        [&callback_called]() { callback_called.store(true, std::memory_order_release); }
    );

    // Don't insert anything, just wait for timeout
    EXPECT_THROW(
        std::ignore = allreduce.wait_and_extract(std::chrono::milliseconds{50}),
        std::runtime_error
    );

    // Callback should not be called since operation never finished
    EXPECT_FALSE(callback_called.load(std::memory_order_acquire));
}

TEST_F(AllReduceFinishedCallbackTest, wait_and_extract_multiple_times) {
    auto this_rank = comm->rank();
    auto nranks = comm->nranks();
    constexpr int n_elements = 5;

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        rapidsmpf::coll::detail::make_host_reduce_operator<int>(SumOp<int>{}),
        br.get(),
        rapidsmpf::Statistics::disabled()
    );

    std::vector<int> data(n_elements);
    for (int j = 0; j < n_elements; j++) {
        data[j] = this_rank;
    }
    auto packed = make_packed(br.get(), data.data(), data.size(), MemoryType::DEVICE);
    allreduce.insert(std::move(packed));

    // First call should succeed
    auto result1 = allreduce.wait_and_extract();
    auto reduced1 = unpack_to_host<int>(result1);
    ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced1.size());
    int const expected_value = (nranks * (nranks - 1)) / 2;
    EXPECT_THAT(reduced1, ::testing::Each(expected_value));

    EXPECT_TRUE(allreduce.finished());

    // Second call: since data was already extracted from underlying AllGather,
    // subsequent calls will get empty data, which should cause reduce_all to throw
    // an error because it expects exactly nranks contributions
    EXPECT_THROW(std::ignore = allreduce.wait_and_extract(), std::runtime_error);
}
