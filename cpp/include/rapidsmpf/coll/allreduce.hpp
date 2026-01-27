/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#endif

#include <cuda_runtime.h>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::coll {

/**
 * @brief Type alias for the reduction function signature.
 *
 * The operator must update the left operand in place by combining it with the right
 * operand. The left operand is passed by lvalue reference and the right operand is
 * passed as an rvalue reference.
 */
using ReduceOperatorFunction = std::function<void(PackedData& left, PackedData&& right)>;

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
 * without requiring runtime checks.
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
     * @param left The left PackedData that will hold the result (updated in place).
     * @param right The right PackedData to combine into the left operand.
     */
    void operator()(PackedData& left, PackedData&& right) const {
        fn(left, std::move(right));
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
 * `detail::make_host_reduce_operator` or
 * `detail::make_device_reduce_operator` can be used to build range-based
 * reductions over contiguous arrays.
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
     * @param reduce_operator Type-erased reduction operator to use. Callers provide a
     * binary operator that acts on the underlying bytes of each element. Use
     * `ReduceOperatorType::Device` for device-side reduction and
     * `ReduceOperatorType::Host` for host-side reduction when constructing the
     * ReduceOperator.
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
    ~AllReduce() = default;

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

    std::atomic<bool> inserted_{false};  ///< Whether insert has been called
};

namespace detail {

/**
 * @brief Host-side range-based reduction operator.
 *
 * This operator applies a binary operation to entire ranges using std::ranges::transform.
 *
 * @tparam T The element type.
 * @tparam Op The binary operation type (e.g., std::plus<T>).
 */
template <typename T, typename Op>
struct HostOp {
    Op op;  ///< The binary reduction operator.

    /**
     * @brief Apply the reduction operator to the packed data ranges.
     *
     * @param left The left PackedData that will be updated in place.
     * @param right The right PackedData to combine into the left operand.
     */
    void operator()(PackedData& left, PackedData&& right) {
        RAPIDSMPF_EXPECTS(
            left.data && right.data, "HostOp requires non-null data buffers"
        );

        auto* left_buf = left.data.get();
        auto* right_buf = right.data.get();

        auto const left_nbytes = left_buf->size;
        auto const right_nbytes = right_buf->size;
        RAPIDSMPF_EXPECTS(
            left_nbytes == right_nbytes, "HostOp requires equal-sized buffers"
        );
        RAPIDSMPF_EXPECTS(
            left_nbytes % sizeof(T) == 0,
            "HostOp buffer size must be a multiple of sizeof(T)"
        );

        auto const count = left_nbytes / sizeof(T);
        if (count == 0) {
            return;
        }

        RAPIDSMPF_EXPECTS(
            left_buf->mem_type() == MemoryType::HOST
                && right_buf->mem_type() == MemoryType::HOST,
            "HostOp expects host memory"
        );

        auto* left_bytes = left_buf->exclusive_data_access();
        auto* right_bytes = right_buf->exclusive_data_access();

        std::span<T> left_span{reinterpret_cast<T*>(left_bytes), count};
        std::span<T const> right_span{reinterpret_cast<T const*>(right_bytes), count};

        std::ranges::transform(left_span, right_span, left_span.begin(), op);

        left_buf->unlock();
        right_buf->unlock();
    }
};

/**
 * @brief Device-side range-based reduction operator.
 *
 * This operator applies a binary operation to entire ranges using thrust::transform.
 *
 * @tparam T The element type.
 * @tparam Op The binary operation type (e.g., cuda::std::plus<T>).
 *
 * @note This struct requires CUDA compilation (__CUDACC__) to be instantiated.
 *       The implementation uses thrust::transform which requires CUDA support.
 */
template <typename T, typename Op>
struct DeviceOp {
    Op op;  ///< The binary reduction operator.

    /**
     * @brief Apply the reduction operator to the packed data ranges.
     *
     * @param left The left PackedData that will be updated in place.
     * @param right The right PackedData to combine into the left operand.
     */
    void operator()(PackedData& left, PackedData&& right) {
#ifdef __CUDACC__
        RAPIDSMPF_EXPECTS(
            left.data && right.data, "DeviceOp requires non-null data buffers"
        );

        auto* left_buf = left.data.get();
        auto* right_buf = right.data.get();

        auto const left_nbytes = left_buf->size;
        auto const right_nbytes = right_buf->size;
        RAPIDSMPF_EXPECTS(
            left_nbytes == right_nbytes, "DeviceOp requires equal-sized buffers"
        );
        RAPIDSMPF_EXPECTS(
            left_nbytes % sizeof(T) == 0,
            "DeviceOp buffer size must be a multiple of sizeof(T)"
        );

        auto const count = left_nbytes / sizeof(T);
        if (count == 0) {
            return;
        }

        RAPIDSMPF_EXPECTS(
            left_buf->mem_type() == MemoryType::DEVICE
                && right_buf->mem_type() == MemoryType::DEVICE,
            "DeviceOp expects device memory"
        );

        cuda_stream_join(left_buf->stream(), right_buf->stream());

        left_buf->write_access([&](std::byte* left_bytes, rmm::cuda_stream_view stream) {
            auto const* right_bytes =
                reinterpret_cast<std::byte const*>(right_buf->data());

            T* left_ptr = reinterpret_cast<T*>(left_bytes);
            T const* right_ptr = reinterpret_cast<T const*>(right_bytes);

            auto policy = thrust::cuda::par_nosync.on(stream.value());
            thrust::transform(
                policy, right_ptr, right_ptr + count, left_ptr, left_ptr, op
            );
        });
#else
        // This should never be reached if DeviceOp is only instantiated with CUDA
        std::ignore = left;
        std::ignore = right;
        RAPIDSMPF_FAIL(
            "DeviceOp::operator() called but CUDA compilation (__CUDACC__) "
            "was not available. DeviceOp requires CUDA/thrust support.",
            std::runtime_error
        );
#endif
    }
};

/**
 * @brief Create a host-based reduction operator from a typed binary operation.
 *
 * @tparam T The element type.
 * @tparam Op The binary operation type.
 * @param op The binary operation (e.g., std::plus<T>{}).
 * @return A ReduceOperator wrapping the HostOp.
 */
template <typename T, typename Op>
    requires std::invocable<Op, T const&, T const&>
ReduceOperator make_host_reduce_operator(Op op) {
    HostOp<T, Op> host_op{std::move(op)};
    return ReduceOperator(
        [host_op = std::move(host_op)](PackedData& left, PackedData&& right) mutable {
            host_op(left, std::move(right));
        },
        ReduceOperatorType::Host
    );
}

/**
 * @brief Create a device-based reduction operator from a typed binary operation.
 *
 * @tparam T The element type.
 * @tparam Op The binary operation type.
 * @param op The binary operation (e.g., cuda::std::plus<T>{}).
 * @return A ReduceOperator wrapping the DeviceOp.
 *
 * @note This function requires CUDA compilation (__CUDACC__ defined) to be used.
 *       Attempting to use it without CUDA will result in a compilation error.
 */
template <typename T, typename Op>
    requires std::invocable<Op, T const&, T const&>
ReduceOperator make_device_reduce_operator(Op op) {
#ifdef __CUDACC__
    DeviceOp<T, Op> device_op{std::move(op)};
    return ReduceOperator(
        [device_op = std::move(device_op)](PackedData& left, PackedData&& right) mutable {
            device_op(left, std::move(right));
        },
        ReduceOperatorType::Device
    );
#else
    std::ignore = op;

    RAPIDSMPF_FAIL(
        "make_device_reduce_operator was called from code that was not compiled "
        "with NVCC (__CUDACC__ is not defined).",
        std::runtime_error
    );
#endif
}

}  // namespace detail

}  // namespace rapidsmpf::coll
