/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
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
T make_input_value(int rank, int elem_idx) {
    // Simple positive pattern valid for all tested arithmetic types.
    auto base = static_cast<std::uint64_t>((rank + 1) * 100 + (elem_idx + 1) * 10);
    return static_cast<T>(base + elem_idx);
}

/**
 * @brief Factory for a host-side reduction kernel for `CustomValue`.
 *
 * The returned kernel expects `PackedData::data` to contain a contiguous array
 * of `CustomValue` in host memory, with equal sizes for all ranks.
 */
ReduceKernel make_custom_value_reduce_kernel_host() {
    return [](PackedData& accum, PackedData&& incoming) {
        RAPIDSMPF_EXPECTS(
            accum.data && incoming.data,
            "CustomValue reduction kernel requires non-null data buffers"
        );

        auto* acc_buf = accum.data.get();
        auto* in_buf = incoming.data.get();

        auto const acc_nbytes = acc_buf->size;
        auto const in_nbytes = in_buf->size;
        RAPIDSMPF_EXPECTS(
            acc_nbytes == in_nbytes,
            "CustomValue reduction kernel requires equal-sized buffers"
        );
        RAPIDSMPF_EXPECTS(
            acc_nbytes % sizeof(CustomValue) == 0,
            "CustomValue reduction kernel requires buffer size to be a multiple "
            "of sizeof(CustomValue)"
        );

        RAPIDSMPF_EXPECTS(
            acc_buf->mem_type() == MemoryType::HOST
                && in_buf->mem_type() == MemoryType::HOST,
            "CustomValue host reduction kernel expects host-backed buffers"
        );

        auto const count = acc_nbytes / sizeof(CustomValue);

        auto* acc_ptr = reinterpret_cast<CustomValue*>(acc_buf->exclusive_data_access());
        auto const* in_ptr = reinterpret_cast<CustomValue const*>(in_buf->data());

        // Perform host-side elementwise reduction
        for (std::size_t i = 0; i < count; ++i) {
            acc_ptr[i].value += in_ptr[i].value;
            acc_ptr[i].weight = std::min(acc_ptr[i].weight, in_ptr[i].weight);
        }

        acc_buf->unlock();
    };
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

    std::vector<int> data(1, 42);  // Simple test data
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
                MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::DEVICE_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_HOST, MemoryReductionConfig::DEVICE_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::HOST_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::MIXED, MemoryReductionConfig::HOST_REDUCTION
            },
            MemoryReductionConfig{
                MemoryReductionConfig::MIXED, MemoryReductionConfig::DEVICE_REDUCTION
            }
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

    // Choose kernel based on reduction type
    ReduceKernel kernel =
        (config.reduction_type == MemoryReductionConfig::DEVICE_REDUCTION)
            ? rapidsmpf::coll::detail::make_device_reduce_kernel<int, ReduceOp::SUM>()
            : rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>();

    bool use_device_reduction =
        (config.reduction_type == MemoryReductionConfig::DEVICE_REDUCTION);

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{0},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        std::move(kernel),
        use_device_reduction
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
    for (int j = 0; j < n_elements; j++) {
        EXPECT_EQ(reduced[static_cast<std::size_t>(j)], expected_value);
    }

    EXPECT_TRUE(allreduce.finished());
}

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

TYPED_TEST(AllReduceTypedOpsTest, basic_allreduce) {
    using Config = TypeParam;
    using T = typename Config::value_type;
    constexpr auto op = Config::reduce_op_id;

    auto this_rank = this->comm->rank();
    auto nranks = this->comm->nranks();

    // Test all combinations of buffer types and reduction types
    std::array<MemoryReductionConfig, 6> configs = {
        {{MemoryReductionConfig::ALL_HOST, MemoryReductionConfig::HOST_REDUCTION},
         {MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::DEVICE_REDUCTION},
         {MemoryReductionConfig::ALL_HOST, MemoryReductionConfig::DEVICE_REDUCTION},
         {MemoryReductionConfig::ALL_DEVICE, MemoryReductionConfig::HOST_REDUCTION},
         {MemoryReductionConfig::MIXED, MemoryReductionConfig::HOST_REDUCTION},
         {MemoryReductionConfig::MIXED, MemoryReductionConfig::DEVICE_REDUCTION}}
    };

    for (auto const& config : configs) {
        ReduceKernel kernel =
            (config.reduction_type == MemoryReductionConfig::DEVICE_REDUCTION)
                ? rapidsmpf::coll::detail::make_device_reduce_kernel<T, op>()
                : rapidsmpf::coll::detail::make_reduce_kernel<T, op>();

        bool use_device_reduction =
            (config.reduction_type == MemoryReductionConfig::DEVICE_REDUCTION);

        AllReduce allreduce(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            OpID{1},
            this->br.get(),
            rapidsmpf::Statistics::disabled(),
            std::move(kernel),
            use_device_reduction
        );

        int constexpr n_elements{8};

        std::vector<T> data(n_elements);
        for (int j = 0; j < n_elements; j++) {
            data[j] = make_input_value<T>(this_rank, j);
        }

        // Choose buffer type based on config
        PackedData packed =
            (config.buffer_type == MemoryReductionConfig::ALL_HOST)
                ? make_packed<T>(
                      this->br.get(), data.data(), data.size(), MemoryType::HOST
                  )
            : (config.buffer_type == MemoryReductionConfig::ALL_DEVICE)
                ? make_packed<T>(
                      this->br.get(), data.data(), data.size(), MemoryType::DEVICE
                  )
            : (this_rank % 2 == 0)  // MIXED: even ranks use host, odd ranks use device
                ? make_packed<T>(
                      this->br.get(), data.data(), data.size(), MemoryType::HOST
                  )
                : make_packed<T>(
                      this->br.get(), data.data(), data.size(), MemoryType::DEVICE
                  );

        allreduce.insert(std::move(packed));
        auto result = allreduce.wait_and_extract();

        auto reduced = unpack_to_host<T>(result);
        ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());
        for (int j = 0; j < n_elements; j++) {
            T expected = make_input_value<T>(0, j);
            for (int r = 1; r < nranks; ++r) {
                expected =
                    Combiner<op>::template apply<T>(expected, make_input_value<T>(r, j));
            }
            if constexpr (std::is_floating_point_v<T>) {
                // For floating point types, use near comparison to account for numerical
                // precision errors
                EXPECT_NEAR(
                    reduced[static_cast<std::size_t>(j)],
                    expected,
                    std::abs(expected) * 1e-5 + 1e-5
                );
            } else {
                EXPECT_EQ(reduced[static_cast<std::size_t>(j)], expected);
            }
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

    int constexpr n_elements{4};

    // Test both host and device custom reduction kernels
    struct TestConfig {
        bool use_device_reduction;
        MemoryType buffer_type;
        const char* name;
    };

    std::array<TestConfig, 2> configs = {
        {{false, MemoryType::HOST, "host_reduction_with_host_buffers"},
         {true, MemoryType::DEVICE, "device_reduction_with_device_buffers"}}
    };

    for (auto const& config : configs) {
        ReduceKernel kernel = config.use_device_reduction
                                  ? make_custom_value_reduce_kernel_device()
                                  : make_custom_value_reduce_kernel_host();

        AllReduce allreduce(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            OpID{5},
            br.get(),
            rapidsmpf::Statistics::disabled(),
            std::move(kernel),
            config.use_device_reduction
        );

        // Each rank contributes the same logical layout of CustomValue
        std::vector<CustomValue> data(n_elements);
        for (int j = 0; j < n_elements; ++j) {
            data[static_cast<std::size_t>(j)].value = make_input_value<int>(this_rank, j);
            data[static_cast<std::size_t>(j)].weight = this_rank * 10 + j;
        }

        auto packed = make_packed<CustomValue>(
            br.get(), data.data(), data.size(), config.buffer_type
        );

        allreduce.insert(std::move(packed));
        auto result = allreduce.wait_and_extract();

        auto reduced = unpack_to_host<CustomValue>(result);
        ASSERT_EQ(static_cast<std::size_t>(n_elements), reduced.size());

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

            auto const& cv = reduced[static_cast<std::size_t>(j)];
            EXPECT_EQ(cv.value, expected_value);
            EXPECT_EQ(cv.weight, expected_weight);
        }
    }
}

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
        OpID{6},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
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
        OpID{7},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>(),
        false,
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
    for (int j = 0; j < n_elements; j++) {
        EXPECT_EQ(reduced[static_cast<std::size_t>(j)], expected_value);
    }

    EXPECT_TRUE(allreduce.finished());
}

TEST_F(AllReduceFinishedCallbackTest, finished_callback_not_called_without_insert) {
    std::atomic<bool> callback_called{false};

    AllReduce allreduce(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        OpID{8},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>(),
        false,
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
        OpID{9},
        br.get(),
        rapidsmpf::Statistics::disabled(),
        rapidsmpf::coll::detail::make_reduce_kernel<int, ReduceOp::SUM>()
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
    for (int j = 0; j < n_elements; j++) {
        EXPECT_EQ(reduced1[static_cast<std::size_t>(j)], expected_value);
    }

    EXPECT_TRUE(allreduce.finished());

    // Second call: since data was already extracted from underlying AllGather,
    // subsequent calls will get empty data, which should cause reduce_all to throw
    // an error because it expects exactly nranks contributions
    EXPECT_THROW(std::ignore = allreduce.wait_and_extract(), std::runtime_error);
}
