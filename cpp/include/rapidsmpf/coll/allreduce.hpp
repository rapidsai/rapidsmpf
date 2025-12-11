/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <ranges>
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
 * @brief Host-side reduction operators supported by `AllReduce`.
 *
 * These closely mirror the reduction operators from MPI.
 */
enum class HostReduceOp : std::uint8_t {
    SUM,  ///< Sum / addition
    PROD,  ///< Product / multiplication
    MIN,  ///< Minimum
    MAX,  ///< Maximum
};

/**
 * @brief Device-side reduction operators supported by `AllReduce`.
 *
 * These closely mirror the reduction operators from MPI.
 */
enum class DeviceReduceOp : std::uint8_t {
    SUM,  ///< Sum / addition
    PROD,  ///< Product / multiplication
    MIN,  ///< Minimum
    MAX,  ///< Maximum
};

/**
 * @brief Type trait to determine if an enum type represents a device operator.
 */
template <typename OpEnum>
struct is_device_reduce_op_enum : std::false_type {};

template <>
struct is_device_reduce_op_enum<DeviceReduceOp> : std::true_type {};

/**
 * @brief Check if an enum type represents a device operator.
 */
template <typename OpEnum>
inline constexpr bool is_device_reduce_op_enum_v =
    is_device_reduce_op_enum<OpEnum>::value;

/**
 * @brief Type alias for the reduction function signature.
 */
using ReduceOperatorFunction =
    std::function<void(PackedData& accum, PackedData&& incoming)>;

/**
 * @brief Enumeration indicating whether a reduction operator runs on host or device.
 */
enum class ReduceOperatorType {
    Host,  ///< Host-side reduction operator
    Device  ///< Device-side reduction operator
};

/**
 * @brief Reduction operator that preserves type information.
 *
 * This class allows AllReduce to determine whether an operator is host- or device-based
 * without requiring runtime checks. Built-in operators use the HostReduceOp and
 * DeviceReduceOp enums to determine this, while custom operators should pass
 * ReduceOperatorType::Device for device-side reduction or ReduceOperatorType::Host for
 * host-side reduction.
 */
class ReduceOperator {
  public:
    /**
     * @brief Construct a new ReduceOperator.
     *
     * @param fn The reduction function to wrap.
     * @param type The type of reduction operator (Host or Device).
     *
     * @throw std::runtime_error if fn is empty/invalid.
     */
    ReduceOperator(ReduceOperatorFunction fn, ReduceOperatorType type)
        : fn(std::move(fn)), type_(type) {
        RAPIDSMPF_EXPECTS(
            static_cast<bool>(this->fn), "ReduceOperator requires a valid function"
        );
    }

    /**
     * @brief Call the wrapped operator.
     *
     * @param accum The accumulator PackedData that will hold the result.
     * @param incoming The incoming PackedData to combine into the accumulator.
     */
    void operator()(PackedData& accum, PackedData&& incoming) const {
        fn(accum, std::move(incoming));
    }

    /**
     * @brief Check if this operator is device-based.
     *
     * @return True if this is a device-based operator, false if host-based.
     */
    [[nodiscard]] bool is_device() const noexcept {
        return type_ == ReduceOperatorType::Device;
    }

  private:
    ReduceOperatorFunction fn;  ///< The reduction function
    ReduceOperatorType type_;  ///< The type of reduction (host or device)
};

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
 * The actual reduction is implemented via a type-erased `ReduceOperator` that is
 * supplied at construction time. Helper factories such as
 * `detail::make_reduce_operator` (defaults to host-side) or
 * `detail::make_device_reduce_operator` (device-side) can be used to build
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
     * @param reduce_operator Type-erased reduction operator to use. For built-in
     * operators, use HostReduceOp and DeviceReduceOp enum values for host- and
     * device-side reduction respectively. For custom operators, pass
     * ReduceOperatorType::Device for device-side reduction or ReduceOperatorType::Host
     * for host-side reduction when constructing the ReduceOperator.
     * @param finished_callback Optional callback run once locally when the allreduce
     * is finished and results are ready for extraction.
     *
     * @note This constructor internally creates an `AllGather` instance
     *       that uses the same communicator, progress thread, and buffer resource.
     */
    AllReduce(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        OpID op_id,
        ReduceOperator reduce_operator,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
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

    ReduceOperator reduce_operator_;  ///< Reduction operator
    BufferResource* br_;  ///< Buffer resource for memory normalization

    Rank nranks_;  ///< Number of ranks in the communicator
    AllGather gatherer_;  ///< Underlying allgather primitive

    bool inserted_{false};  ///< Whether insert has been called
};

namespace detail {

/// Valid HostReduceOp values.
inline constexpr std::array valid_host_reduce_ops = {
    HostReduceOp::SUM, HostReduceOp::PROD, HostReduceOp::MIN, HostReduceOp::MAX
};

/// Valid DeviceReduceOp values.
inline constexpr std::array valid_device_reduce_ops = {
    DeviceReduceOp::SUM, DeviceReduceOp::PROD, DeviceReduceOp::MIN, DeviceReduceOp::MAX
};

/**
 * @brief Concept to validate (T, HostReduceOp) combinations for reduction.
 */
template <typename T, HostReduceOp Op>
concept ValidHostReduceOp =
    std::is_arithmetic_v<T>
    && std::ranges::find(valid_host_reduce_ops, Op) != valid_host_reduce_ops.end();

/**
 * @brief Concept to validate (T, DeviceReduceOp) combinations for reduction.
 */
template <typename T, DeviceReduceOp Op>
concept ValidDeviceReduceOp =
    std::is_arithmetic_v<T>
    && std::ranges::find(valid_device_reduce_ops, Op) != valid_device_reduce_ops.end();

/**
 * @brief Concept for a binary operator on type T.
 */
template <typename Op, typename T>
concept BinaryOp =
    std::invocable<Op, T const&, T const&>
    && std::convertible_to<std::invoke_result_t<Op, T const&, T const&>, T>;

template <typename T, typename Op>
    requires BinaryOp<Op, T>
void apply_op(PackedData& accum, PackedData&& incoming, Op&& op) {
    RAPIDSMPF_EXPECTS(
        accum.data && incoming.data,
        "AllReduce reduction operator requires non-null data buffers"
    );

    auto* acc_buf = accum.data.get();
    auto* in_buf = incoming.data.get();

    auto const acc_nbytes = acc_buf->size;
    auto const in_nbytes = in_buf->size;
    RAPIDSMPF_EXPECTS(
        acc_nbytes == in_nbytes,
        "AllReduce reduction operator requires equal-sized buffers"
    );

    auto const nbytes = acc_nbytes;
    RAPIDSMPF_EXPECTS(
        nbytes % sizeof(T) == 0,
        "AllReduce reduction operator requires buffer size to be a multiple of "
        "sizeof(T)"
    );

    auto const count = nbytes / sizeof(T);

    RAPIDSMPF_EXPECTS(
        acc_buf->mem_type() == MemoryType::HOST && in_buf->mem_type() == MemoryType::HOST,
        "make_host_reduce_operator expects host memory"
    );

    auto* acc_bytes = acc_buf->exclusive_data_access();
    auto* in_bytes = in_buf->exclusive_data_access();
    auto* acc_ptr = reinterpret_cast<T*>(acc_bytes);
    auto const* in_ptr = reinterpret_cast<T const*>(in_bytes);

    std::transform(acc_ptr, acc_ptr + count, in_ptr, acc_ptr, op);

    acc_buf->unlock();
    in_buf->unlock();
}

/**
 * @brief Create a functor for a given (T, HostReduceOp) combination.
 */
template <typename T, HostReduceOp Op>
    requires ValidHostReduceOp<T, Op>
constexpr auto make_functor() {
    if constexpr (Op == HostReduceOp::SUM) {
        if constexpr (std::is_same_v<T, bool>) {
            return [](T const& a, T const& b) -> T { return static_cast<T>(a || b); };
        }
        return std::plus<T>{};
    } else if constexpr (Op == HostReduceOp::PROD) {
        return std::multiplies<T>{};
    } else if constexpr (Op == HostReduceOp::MIN) {
        return [](T const& a, T const& b) -> T { return std::min(a, b); };
    } else if constexpr (Op == HostReduceOp::MAX) {
        return [](T const& a, T const& b) -> T { return std::max(a, b); };
    }
}

/**
 * @brief Create a host-based element-wise reduction operator wrapper for a given (T, Op).
 */
template <typename T, HostReduceOp Op>
    requires ValidHostReduceOp<T, Op>
ReduceOperator make_host_reduce_operator() {
    return ReduceOperator(
        [](PackedData& accum, PackedData&& incoming) {
            apply_op<T>(accum, std::move(incoming), make_functor<T, Op>());
        },
        ReduceOperatorType::Host
    );
}

/**
 * @brief Create a device-based element-wise reduction operator implementation for a given
 * (T, Op).
 *
 * This operator expects both `PackedData::data` buffers to reside in device memory.
 * Implementations are provided in `device_kernels.cu` for a subset of (T, Op)
 * combinations.
 */
template <typename T, DeviceReduceOp Op>
    requires ValidDeviceReduceOp<T, Op>
ReduceOperatorFunction make_device_reduce_operator_impl();

/**
 * @brief Create a device-based element-wise reduction operator wrapper for a given (T,
 * Op).
 */
template <typename T, DeviceReduceOp Op>
    requires ValidDeviceReduceOp<T, Op>
ReduceOperator make_device_reduce_operator() {
    return ReduceOperator(
        make_device_reduce_operator_impl<T, Op>(), ReduceOperatorType::Device
    );
}

/**
 * @brief Create a reduction operator for a given (T, HostReduceOp).
 *
 * Convenience overload for host-side operators.
 */
template <typename T, HostReduceOp Op>
    requires ValidHostReduceOp<T, Op>
ReduceOperator make_reduce_operator() {
    return make_host_reduce_operator<T, Op>();
}

/**
 * @brief Create a reduction operator for a given (T, DeviceReduceOp).
 *
 * Convenience overload for device-side operators.
 */
template <typename T, DeviceReduceOp Op>
    requires ValidDeviceReduceOp<T, Op>
ReduceOperator make_reduce_operator() {
    return make_device_reduce_operator<T, Op>();
}

}  // namespace detail

}  // namespace rapidsmpf::coll
