/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::coll {

/**
 * @brief Reduction operators supported by `AllReduce`.
 *
 * These closely mirror the reduction operators from MPI.
 */
enum class ReduceOp : std::uint8_t {
    SUM,  ///< Sum / addition
    PROD,  ///< Product / multiplication
    MIN,  ///< Minimum
    MAX,  ///< Maximum
};

/**
 * @brief Type-erased reduction kernel used by `AllReduce`.
 *
 * The kernel must implement an associative binary operation over the contents of
 * two `PackedData` objects and accumulate the result into @p accum.
 *
 * Implementations must:
 *  - Treat @p accum as the running partial result.
 *  - Combine @p incoming into @p accum in-place.
 *  - Leave @p incoming in a valid but unspecified state after the call.
 *
 * The kernel is responsible for interpreting `PackedData::metadata` and
 * `PackedData::data` consistently across all ranks.
 */
using ReduceKernel = std::function<void(PackedData& accum, PackedData&& incoming)>;

/**
 * @brief AllReduce collective.
 *
 * The current implementation is built using `coll::AllGather` and performs
 * the reduction locally after allgather completes. Considering `R` is the number of
 * ranks, and `N` is the number of bytes of data, per rank this incurs `O(R * N)` bytes of
 * memory consumption and `O(R)` communication operations.
 *
 * Semantics:
 *  - Each rank calls `insert` exactly once to contribute data to the reduction.
 *  - Once all ranks call `insert`, `wait_and_extract` returns the
 *    globally-reduced `PackedData`.
 *
 * The actual reduction is implemented via a type-erased `ReduceKernel` that is
 * supplied at construction time. Helper factories such as
 * `detail::make_reduction_kernel` (defaults to host-side) or
 * `detail::make_device_reduce_kernel` (device-side) can be used to build
 * element-wise reductions over contiguous arrays.
 */
class AllReduce {
  public:
    /**
     * @brief Construct a new AllReduce operation.
     *
     * @param comm The communicator for communication.
     * @param progress_thread The progress thread used by the underlying AllGather.
     * @param op_id Unique operation identifier for this allreduce.
     * @param br Buffer resource for memory allocation.
     * @param statistics Statistics collection instance (disabled by default).
     * @param reduce_kernel Type-erased reduction kernel to use.
     * @param use_device_reduction If true, perform reduction on device memory.
     *        If false (default), perform reduction on host memory. Buffers will
     *        be normalized to the target memory type before reduction.
     * @param finished_callback Optional callback run once locally when the allreduce
     *        is finished and results are ready for extraction.
     *
     * @note This constructor internally creates an `AllGather` instance
     *       that uses the same communicator, progress thread, and buffer resource.
     */
    AllReduce(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        OpID op_id,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        ReduceKernel reduce_kernel = {},
        bool use_device_reduction = false,
        std::function<void(void)> finished_callback = nullptr
    );

    AllReduce(AllReduce const&) = delete;
    AllReduce& operator=(AllReduce const&) = delete;
    AllReduce(AllReduce&&) = delete;
    AllReduce& operator=(AllReduce&&) = delete;

    /**
     * @brief Destructor.
     *
     * @note This operation is logically collective. If an `AllReduce` is locally
     *       destructed before `wait_and_extract` is called, there is no guarantee
     *       that in-flight communication will be completed.
     */
    ~AllReduce();

    /**
     * @brief Insert packed data into the allreduce operation.
     *
     * @param packed_data The data to contribute to the allreduce.
     *
     * @throws std::runtime_error If insert has already been called on this instance.
     */
    void insert(PackedData&& packed_data);

    /**
     * @brief Check if the allreduce operation has completed.
     *
     * @return True if all data and finish messages from all ranks have been
     *         received and locally reduced.
     */
    [[nodiscard]] bool finished() const noexcept;

    /**
     * @brief Wait for completion and extract the reduced data.
     *
     * Blocks until the allreduce operation completes and returns the
     * globally reduced result.
     *
     * @param timeout Optional maximum duration to wait. Negative values mean
     *        no timeout.
     *
     * @return The reduced packed data.
     * @throws std::runtime_error If the timeout is reached.
     */
    [[nodiscard]] PackedData wait_and_extract(
        std::chrono::milliseconds timeout = std::chrono::milliseconds{-1}
    );

    /**
     * @brief Check if reduced results are ready for extraction.
     *
     * This returns true once the underlying allgather has completed and, if
     * `wait_and_extract` has not yet been called, indicates that calling it
     * would not block.
     *
     * @return True if the allreduce operation has completed and results are ready for
     *         extraction, false otherwise.
     */
    [[nodiscard]] bool is_ready() const noexcept;

  private:
    /// @brief Perform the reduction across all ranks for the gathered contributions.
    [[nodiscard]] PackedData reduce_all(std::vector<PackedData>&& gathered);

    BufferResource* br_;  ///< Buffer resource for memory normalization
    ReduceKernel reduce_kernel_;  ///< Type-erased reduction kernel
    bool use_device_reduction_;  ///< Whether to perform reduction on device memory

    Rank nranks_;  ///< Number of ranks in the communicator
    AllGather gatherer_;  ///< Underlying allgather primitive

    bool inserted_{false};  ///< Whether insert has been called
};

namespace detail {

/**
 * @brief Type trait to check if a type supports arithmetic operations.
 */
template <typename T>
struct is_supported_arithmetic_op : std::false_type {};

template <>
struct is_supported_arithmetic_op<int> : std::true_type {};

template <>
struct is_supported_arithmetic_op<float> : std::true_type {};

template <>
struct is_supported_arithmetic_op<double> : std::true_type {};

template <>
struct is_supported_arithmetic_op<unsigned long> : std::true_type {};

template <>
struct is_supported_arithmetic_op<bool> : std::true_type {};

/**
 * @brief Type trait to check if a (T, Op) combination is supported.
 */
template <typename T, ReduceOp Op>
struct is_supported_reduce_op : std::false_type {};

template <typename T>
struct is_supported_reduce_op<T, ReduceOp::SUM> : is_supported_arithmetic_op<T> {};

template <typename T>
struct is_supported_reduce_op<T, ReduceOp::PROD> : is_supported_arithmetic_op<T> {};

template <typename T>
struct is_supported_reduce_op<T, ReduceOp::MIN> : is_supported_arithmetic_op<T> {};

template <typename T>
struct is_supported_reduce_op<T, ReduceOp::MAX> : is_supported_arithmetic_op<T> {};

template <typename T, ReduceOp Op>
inline constexpr bool is_supported_reduce_op_v = is_supported_reduce_op<T, Op>::value;

/**
 * @brief Create a host-based element-wise reduction kernel for a given (T, Op).
 *
 * The kernel assumes that the `PackedData::data` buffers reside in host memory
 * and contain a contiguous array of @p T with identical sizes on all ranks.
 */
template <typename T, ReduceOp Op>
ReduceKernel make_host_reduce_kernel() {
    static_assert(
        is_supported_reduce_op_v<T, Op>,
        "make_host_reduce_kernel called for unsupported (T, Op) combination"
    );

    auto const apply = [](PackedData& accum, PackedData&& incoming, auto&& op) {
        RAPIDSMPF_EXPECTS(
            accum.data && incoming.data,
            "AllReduce reduction kernel requires non-null data buffers"
        );

        auto* acc_buf = accum.data.get();
        auto* in_buf = incoming.data.get();

        auto const acc_nbytes = acc_buf->size;
        auto const in_nbytes = in_buf->size;
        RAPIDSMPF_EXPECTS(
            acc_nbytes == in_nbytes,
            "AllReduce reduction kernel requires equal-sized buffers"
        );

        auto const nbytes = acc_nbytes;
        RAPIDSMPF_EXPECTS(
            nbytes % sizeof(T) == 0,
            "AllReduce reduction kernel requires buffer size to be a multiple of "
            "sizeof(T)"
        );

        auto const count = nbytes / sizeof(T);

        RAPIDSMPF_EXPECTS(
            acc_buf->mem_type() == MemoryType::HOST
                && in_buf->mem_type() == MemoryType::HOST,
            "make_host_reduce_kernel expects host memory"
        );

        auto* acc_bytes = acc_buf->exclusive_data_access();
        auto* in_bytes = in_buf->exclusive_data_access();
        auto* acc_ptr = reinterpret_cast<T*>(acc_bytes);
        auto const* in_ptr = reinterpret_cast<T const*>(in_bytes);

        for (std::size_t i = 0; i < count; ++i) {
            acc_ptr[i] = op(acc_ptr[i], in_ptr[i]);
        }

        acc_buf->unlock();
        in_buf->unlock();
    };

    if constexpr (Op == ReduceOp::SUM) {
        return [apply](PackedData& accum, PackedData&& incoming) {
            if constexpr (std::is_same_v<T, bool>) {
                apply(accum, std::move(incoming), [](T a, T b) {
                    return static_cast<T>(a || b);
                });
            } else {
                apply(accum, std::move(incoming), std::plus<T>{});
            }
        };
    } else if constexpr (Op == ReduceOp::PROD) {
        return [apply](PackedData& accum, PackedData&& incoming) {
            apply(accum, std::move(incoming), std::multiplies<T>{});
        };
    } else if constexpr (Op == ReduceOp::MIN) {
        return [apply](PackedData& accum, PackedData&& incoming) {
            apply(accum, std::move(incoming), [](T a, T b) { return std::min(a, b); });
        };
    } else if constexpr (Op == ReduceOp::MAX) {
        return [apply](PackedData& accum, PackedData&& incoming) {
            apply(accum, std::move(incoming), [](T a, T b) { return std::max(a, b); });
        };
    } else {
        static_assert(
            Op == ReduceOp::SUM || Op == ReduceOp::PROD || Op == ReduceOp::MIN
                || Op == ReduceOp::MAX,
            "AllReduce kernel only implemented for SUM, PROD, MIN, and MAX"
        );
    }
}

/**
 * @brief Create a device-based element-wise reduction kernel for a given (T, Op).
 *
 * This kernel expects both `PackedData::data` buffers to reside in device memory.
 * Implementations are provided in `device_kernels.cu` for a subset of (T, Op)
 * combinations.
 */
template <typename T, ReduceOp Op>
ReduceKernel make_device_reduce_kernel();

/**
 * @brief Create a memory-type aware reduction kernel for a given (T, Op).
 *
 * The returned kernel defaults to host-side reduction. If all buffers are in device
 * memory and device reduction is explicitly requested, it will use device reduction.
 * Otherwise, it normalizes all buffers to host memory and performs host-side reduction.
 *
 * @param prefer_device If true, prefer device-side reduction when all buffers are
 *        device-resident. If false (default), always use host-side reduction.
 */
template <typename T, ReduceOp Op>
ReduceKernel make_reduction_kernel(bool prefer_device = false) {
    auto host_kernel = make_host_reduce_kernel<T, Op>();
    auto device_kernel = make_device_reduce_kernel<T, Op>();

    return [host = std::move(host_kernel),
            device = std::move(device_kernel),
            prefer_device](PackedData& accum, PackedData&& incoming) mutable {
        RAPIDSMPF_EXPECTS(
            accum.data && incoming.data,
            "AllReduce reduction kernel requires data buffers"
        );

        auto const mem_type = accum.data->mem_type();

        if (prefer_device && mem_type == MemoryType::DEVICE) {
            device(accum, std::move(incoming));
        } else {
            // Default to host kernel for HOST (and any other non-DEVICE types such as
            // MANAGED).
            host(accum, std::move(incoming));
        }
    };
}

}  // namespace detail

}  // namespace rapidsmpf::coll
