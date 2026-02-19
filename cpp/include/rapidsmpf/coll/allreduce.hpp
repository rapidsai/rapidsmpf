/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

#ifdef __CUDACC__
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#endif

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::coll {

/**
 * @brief Type alias for the reduction function signature.
 *
 * A reduction function is a binary operator `left \oplus right`. The function
 * implementing the operation must update `right` in place. That is, the result of calling
 * the reduction should be as if we do `right <- left \oplus right`.
 */
using ReduceOperator = std::function<void(Buffer const* left, Buffer* right)>;

/**
 * @brief AllReduce collective.
 *
 * The implementation uses a butterfly recursive doubling scheme for message exchange,
 * using no extra memory and `O(log P)` rounds for `P` ranks.
 *
 * The actual reduction is implemented via a type-erased `ReduceOperator` that is supplied
 * at construction time. Helper factories such as `detail::make_host_reduce_operator` or
 * `detail::make_device_reduce_operator` can be used to build range-based reductions over
 * contiguous arrays.
 *
 * @note No internal allocations are made. The memory types and sizes of the two provided
 * buffers must match, and the provided reduction operator must be valid for the memory
 * type of the buffers.
 *
 * @note The reduction is safe to use with both non-associative and non-commutative
 * reduction operations in the sense that all participating ranks are guaranteed to
 * receive the same answer even if the operator is not associative or commutative.
 *
 * @note It is safe to reuse the `op_id` passed to the `AllReduce` construction locally as
 * soon as `wait_and_extract` is complete.
 */
class AllReduce {
  public:
    /**
     * @brief Construct a new AllReduce operation.
     *
     * @param comm The communicator for communication.
     * @param progress_thread The progress thread used by the underlying AllGather.
     * @param input Local data to contribute to the reduction.
     * @param output Allocated buffer in which to place reduction result. Must be the same
     * size and memory type as `input`.
     * @param op_id Unique operation identifier for this allreduce.
     * @param reduce_operator Type-erased reduction operator to use. See `ReduceOperator`.
     * @param finished_callback Optional callback run once locally when the allreduce
     * is finished and results are ready for extraction.
     *
     * @throws std::invalid_argument If the input and output buffers do not match
     * appropriately (same size, same memory type).
     */
    AllReduce(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        std::unique_ptr<Buffer> input,
        std::unique_ptr<Buffer> output,
        OpID op_id,
        ReduceOperator reduce_operator,
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
     * destructed before `wait_and_extract` is called, there is no guarantee
     * that in-flight communication will be completed.
     */
    ~AllReduce() noexcept;

    /**
     * @brief Check if the allreduce operation has completed.
     *
     * @return True if all data and finish messages from all ranks have been
     * received and locally reduced.
     */
    [[nodiscard]] bool finished() const noexcept;

    /**
     * @brief Wait for completion and extract the reduced data.
     *
     * Blocks until the allreduce operation completes and returns the globally reduced
     * result.
     *
     * This method is destructive and can only be called once. The first call extracts the
     * buffers provided to the `AllReduce` constructor. Subsequent calls will throw
     * std::runtime_error because the underlying data has already been consumed.
     *
     * @param timeout Optional maximum duration to wait. Negative values mean
     * no timeout.
     * @return A pair of the two `Buffer`s passed to the constructor. The first `Buffer`
     * contains an implementation-defined value, the second `Buffer` contains the
     * final reduced result.
     *
     * @note The streams of the Buffers may change in an implementation-defined way while
     * owned by the `AllReduce` object, if you need to launch new stream-ordered work on a
     * `Buffer` you obtain from this function, you _must_ obtain the correct stream from
     * the `Buffer` itself.
     *
     * @throws std::runtime_error If the timeout is reached or if this method is
     * called more than once.
     */
    [[nodiscard]] std::pair<std::unique_ptr<Buffer>, std::unique_ptr<Buffer>>
    wait_and_extract(std::chrono::milliseconds timeout = std::chrono::milliseconds{-1});

    /**
     * @brief Check if reduced results are ready for extraction.
     *
     * This returns true once the underlying allgather has completed and, if
     * `wait_and_extract` has not yet been called, indicates that calling it
     * would not block.
     *
     * @return True if the allreduce operation has completed and results are ready for
     * extraction, false otherwise.
     */
    [[nodiscard]] bool is_ready() const noexcept;

  private:
    enum class Phase : uint8_t {
        StartPreRemainder,
        CompletePreRemainder,
        StartButterfly,
        CompleteButterfly,
        StartPostRemainder,
        CompletePostRemainder,
        Done,
        ResultAvailable
    };

    /// @brief Progress the non-blocking allreduce state machine.
    [[nodiscard]] ProgressThread::ProgressState event_loop();

    std::shared_ptr<Communicator> comm_{};
    std::shared_ptr<ProgressThread> progress_thread_{};
    ReduceOperator reduce_operator_;  ///< Reduction operator
    std::unique_ptr<Buffer> in_buffer_{};
    std::unique_ptr<Buffer> out_buffer_{};
    OpID op_id_{};
    std::atomic<Phase> phase_{Phase::StartPreRemainder};
    std::atomic<bool> active_{true};
    std::function<void()>
        finished_callback_;  ///< Callback invoked when allreduce completes

    mutable std::mutex mutex_;  ///< Mutex for synchronization
    mutable std::condition_variable cv_;  ///< Condition variable for waiting

    Rank logical_rank_{-1};
    Rank nearest_pow2_{0};
    Rank non_pow2_remainder_{0};
    Rank stage_mask_{1};
    Rank stage_partner_{-1};

    ProgressThread::FunctionID function_id_{};  ///< Progress thread function id

    std::unique_ptr<Communicator::Future> send_future_{};
    std::unique_ptr<Communicator::Future> recv_future_{};
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
     * @param left The left Buffer that will be combined.
     * @param right The right Buffer updated with the left operand.
     */
    void operator()(Buffer const* left, Buffer* right) {
        auto const left_nbytes = left->size;
        RAPIDSMPF_EXPECTS(
            left_nbytes % sizeof(T) == 0,
            "HostOp buffer size must be a multiple of sizeof(T)"
        );

        auto const count = left_nbytes / sizeof(T);
        if (count == 0) {
            return;
        }

        RAPIDSMPF_EXPECTS(
            left->mem_type() == MemoryType::HOST && right->mem_type() == MemoryType::HOST,
            "HostOp expects host memory"
        );

        auto* left_bytes = left->data();
        auto* right_bytes = right->exclusive_data_access();

        std::span<T const> left_span{reinterpret_cast<T const*>(left_bytes), count};
        std::span<T> right_span{reinterpret_cast<T*>(right_bytes), count};

        std::ranges::transform(left_span, right_span, right_span.begin(), op);
        right->unlock();
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
     * @param left The left Buffer that will be combined.
     * @param right The right Buffer updated with the left operand.
     */
    void operator()(Buffer const* left, Buffer* right) {
#ifdef __CUDACC__
        auto const left_nbytes = left->size;
        RAPIDSMPF_EXPECTS(
            left_nbytes % sizeof(T) == 0,
            "DeviceOp buffer size must be a multiple of sizeof(T)"
        );

        auto const count = left_nbytes / sizeof(T);
        if (count == 0) {
            return;
        }

        RAPIDSMPF_EXPECTS(
            left->mem_type() == MemoryType::DEVICE
                && right->mem_type() == MemoryType::DEVICE,
            "DeviceOp expects device memory"
        );
        // Both buffers are guaranteed to be on the same stream by the insertion
        // implementation.
        right->write_access([&](std::byte* right_bytes, rmm::cuda_stream_view stream) {
            auto const* left_bytes = reinterpret_cast<std::byte const*>(left->data());

            T* right_ptr = reinterpret_cast<T*>(right_bytes);
            T const* left_ptr = reinterpret_cast<T const*>(left_bytes);

            thrust::transform(
                thrust::cuda::par_nosync.on(stream.value()),
                left_ptr,
                left_ptr + count,
                right_ptr,
                right_ptr,
                op
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
    return [host_op = std::move(host_op)](Buffer const* left, Buffer* right) mutable {
        host_op(left, right);
    };
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
 * Attempting to use it without CUDA will result in a compilation error.
 */
template <typename T, typename Op>
    requires std::invocable<Op, T const&, T const&>
ReduceOperator make_device_reduce_operator(Op op) {
#ifdef __CUDACC__
    DeviceOp<T, Op> device_op{std::move(op)};
    return [device_op = std::move(device_op)](Buffer const* left, Buffer* right) mutable {
        device_op(left, right);
    };
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
