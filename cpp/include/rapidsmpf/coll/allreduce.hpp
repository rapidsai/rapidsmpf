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
 * `detail::make_host_reduce_operator` or
 * `detail::make_device_reduce_operator` can be used to build element-wise
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

/**
 * @brief Host-side elementwise operator working on raw bytes.
 *
 * The callable receives pointers to a single element of `accum` and `incoming`
 * respectively and must combine them in-place into `accum`. The caller
 * guarantees that both pointers reference `element_size` bytes.
 */
using HostByteOp = std::function<void(void* accum_elem, void const* incoming_elem)>;

/**
 * @brief Apply a host-based byte-wise reduction operator to the given packed data.
 *
 * @param accum The accumulator packed data.
 * @param incoming The incoming packed data.
 * @param element_size Size of each element in bytes.
 * @param op The host-based byte-wise reduction operator.
 */
inline void apply_host_byte_op(
    PackedData& accum,
    PackedData&& incoming,
    std::size_t element_size,
    HostByteOp const& op
) {
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
        element_size > 0 && nbytes % element_size == 0,
        "AllReduce reduction operator requires buffer size to be a multiple of "
        "element_size"
    );

    auto const count = nbytes / element_size;

    RAPIDSMPF_EXPECTS(
        acc_buf->mem_type() == MemoryType::HOST && in_buf->mem_type() == MemoryType::HOST,
        "Host reduction operator expects host memory"
    );

    auto* acc_bytes = acc_buf->exclusive_data_access();
    auto* in_bytes = in_buf->exclusive_data_access();

    for (std::size_t i = 0; i < count; ++i) {
        auto* acc_elem = static_cast<void*>(acc_bytes + i * element_size);
        auto const* in_elem = static_cast<void const*>(in_bytes + i * element_size);
        op(acc_elem, in_elem);
    }

    acc_buf->unlock();
    in_buf->unlock();
}

/**
 * @brief Create a host-based element-wise reduction operator wrapper using a byte-wise
 * op.
 *
 * @param element_size Size of each element in bytes.
 * @param op Byte-wise operator invoked per element.
 *
 * @return The wrapped reduction operator.
 */
inline ReduceOperator make_host_byte_reduce_operator(
    std::size_t element_size, HostByteOp op
) {
    RAPIDSMPF_EXPECTS(
        element_size > 0, "Host reduction operator requires element_size>0"
    );
    RAPIDSMPF_EXPECTS(
        static_cast<bool>(op), "Host reduction operator requires a callable"
    );

    return ReduceOperator(
        [element_size, op = std::move(op)](
            PackedData& accum, PackedData&& incoming
        ) mutable { apply_host_byte_op(accum, std::move(incoming), element_size, op); },
        ReduceOperatorType::Host
    );
}

/**
 * @brief Create a host-based element-wise reduction operator wrapper for a typed functor.
 */
template <typename T, typename Op>
    requires std::invocable<Op, T const&, T const&>
ReduceOperator make_host_reduce_operator(Op op) {
    /**
     * @brief Create a host-based element-wise reduction operator wrapper for a typed
     * functor.
     *
     * @param op The reduction operator to wrap.
     *
     * @return The wrapped reduction operator.
     */
    return make_host_byte_reduce_operator(
        sizeof(T), [op = std::move(op)](void* accum_elem, void const* incoming_elem) {
            auto* a = reinterpret_cast<T*>(accum_elem);
            auto const* b = reinterpret_cast<T const*>(incoming_elem);
            *a = static_cast<T>(std::invoke(op, *a, *b));
        }
    );
}

namespace device {

template <typename DeviceOp>
ReduceOperatorFunction make_device_byte_reduce_operator(
    std::size_t element_size, DeviceOp op
);

}  // namespace device

template <typename DeviceOp>
ReduceOperator make_device_byte_reduce_operator(std::size_t element_size, DeviceOp op) {
    /**
     * @brief Create a device-based element-wise reduction operator wrapper using a
     * byte-wise op.
     *
     * @param element_size Size of each element in bytes.
     * @param op Byte-wise operator invoked per element.
     *
     * @return The wrapped reduction operator.
     */
    return ReduceOperator(
        device::make_device_byte_reduce_operator(element_size, std::move(op)),
        ReduceOperatorType::Device
    );
}

/**
 * @brief Device-side element-wise reduction operator wrapper for a typed functor.
 */
#ifdef __CUDACC__
#define RAPIDSMPF_HD __host__ __device__
#else
#define RAPIDSMPF_HD
#endif

/**
 * @brief Device-side element-wise reduction operator wrapper for a typed functor.
 */
template <typename T, typename Op>
struct DeviceElementwiseOp {
    Op op;  ///< The reduction function to wrap.

    /**
     * @brief Call the wrapped operator.
     *
     * @param accum_elem The accumulator element.
     * @param incoming_elem The incoming element.
     */
    RAPIDSMPF_HD void operator()(void* accum_elem, void const* incoming_elem) const {
        auto* a = reinterpret_cast<T*>(accum_elem);
        auto const* b = reinterpret_cast<T const*>(incoming_elem);
        *a = static_cast<T>(std::invoke(op, *a, *b));
    }
};

#undef RAPIDSMPF_HD

/**
 * @brief Create a device-based element-wise reduction operator wrapper for a typed
 * functor.
 *
 * @param op The reduction operator to wrap.
 *
 * @return The wrapped reduction operator.
 */
template <typename T, typename Op>
ReduceOperator make_device_reduce_operator(Op op) {
    return make_device_byte_reduce_operator(
        sizeof(T), DeviceElementwiseOp<T, Op>{std::move(op)}
    );
}

}  // namespace detail

}  // namespace rapidsmpf::coll
