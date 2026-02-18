/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <thread>

#include <coro/coro.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_executor.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Context for nodes (coroutines) in rapidsmpf.
 *
 * The context owns shared resources used during execution, including the
 * coroutine executor and memory reservation infrastructure.
 *
 * @warning Shutdown of the context must be initiated from the same thread that
 * created it. Calling `shutdown()` from a different thread results in program
 * termination. Since the destructor implicitly calls `shutdown()`, destroying
 * the context from a different thread also results in termination unless the
 * executor has already been shut down explicitly.
 *
 * A recommended usage pattern is to create a single `Context` instance up front
 * on the main thread and reuse it throughout the lifetime of the program. This
 * reduces overhead and avoids issues related to destruction on a different
 * thread.
 */
class Context {
  public:
    /**
     * @brief Full constructor for the Context.
     *
     * All provided pointers must be non-null.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param progress_thread Shared pointer to a progress thread.
     * @param executor Shared pointer to a coroutine executor.
     * @param br Shared pointer to a buffer resource.
     * @param statistics Shared pointer to a statistics collector.
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        std::shared_ptr<CoroThreadPoolExecutor> executor,
        std::shared_ptr<BufferResource> br,
        std::shared_ptr<Statistics> statistics
    );

    /**
     * @brief Convenience constructor using the provided configuration options.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param br Buffer resource used to reserve host memory and perform data movement.
     * @param statistics Statistics instance to use (disabled by default).
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<BufferResource> br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Create a Context based on configuration options.
     *
     * This is a convenience factory that wires up a fully initialized and usable
     * Context.
     *
     * @note The current CUDA device must be set prior to calling this function.
     * Options that depend on device memory availability query the current device.
     *
     * @param mr Device memory resource adaptor used by RapidsMPF. The adaptor must
     * outlive the returned Context.
     * @param comm The communicator to use.
     * @param options Configuration options used to initialize the Context and its
     * components.
     * @return A fully initialized Context.
     *
     * @throws std::invalid_argument If an option value is invalid.
     * @throws std::out_of_range If an option value exceeds the representable range.
     *
     * @warning Shutdown of the context must be initiated from the same thread that
     * created it. Calling `shutdown()` from a different thread results in program
     * termination. Since the destructor implicitly calls `shutdown()`, destroying
     * the context from a different thread also results in termination unless the
     * executor has already been shut down explicitly.
     *
     * A recommended usage pattern is to create a single `Context` instance up front
     * on the main thread and reuse it throughout the lifetime of the program. This
     * reduces overhead and avoids issues related to destruction on a different
     * thread.
     */
    static std::shared_ptr<Context> from_options(
        RmmResourceAdaptor* mr,
        std::shared_ptr<Communicator> comm,
        config::Options options
    );

    // No copy constructor and assignment operator.
    Context(Context const&) = delete;
    Context& operator=(Context const&) = delete;

    // No move constructor and assignment operator.
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    ~Context() noexcept;

    /**
     * @brief Shut down the context.
     *
     * This method is idempotent and only performs shutdown once. Subsequent calls
     * have no effect.
     *
     * @warning Shutdown must be initiated from the same thread that constructed
     * the executor. Calling this method from a different thread results in program
     * termination.
     */
    void shutdown() noexcept;

    /**
     * @brief Returns the configuration options.
     *
     * @return The Options instance.
     */
    [[nodiscard]] config::Options options() const noexcept;

    /**
     * @brief Returns the communicator.
     *
     * @return Shared pointer to the communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> comm() const noexcept;

    /**
     * @brief Returns the logger.
     *
     * @return Reference to the logger.
     */
    [[nodiscard]] Communicator::Logger& logger() const noexcept;

    /**
     * @brief Returns the progress thread.
     *
     * @return Shared pointer to the progress thread.
     */
    [[nodiscard]] std::shared_ptr<ProgressThread> progress_thread() const noexcept;

    /**
     * @brief Returns the coroutine executor.
     *
     * @return Reference to unique pointer to the executor.
     */
    [[nodiscard]] std::shared_ptr<CoroThreadPoolExecutor> executor() const noexcept;

    /**
     * @brief Returns the buffer resource.
     *
     * @return Raw pointer to the buffer resource.
     */
    [[nodiscard]] std::shared_ptr<BufferResource> br() const noexcept;

    /**
     * @brief Get the handle for memory reservations for a given memory type.
     *
     * Returns an object that coordinates asynchronous memory reservation requests
     * for the specified memory type. The returned instance provides backpressure
     * and global progress guarantees, and should be used to reserve memory before
     * performing operations that require memory.
     *
     * A recommended usage pattern is to reserve all required memory up front as a
     * single atomic reservation. This allows callers to `co_await` the reservation
     * request and only start executing the operation once all required memory is
     * available.
     *
     * @param mem_type Memory type for which reservations are requested.
     * @return Shared pointer to the corresponding memory reservation coordinator.
     */
    [[nodiscard]] std::shared_ptr<MemoryReserveOrWait> memory(
        MemoryType mem_type
    ) const noexcept;

    /**
     * @brief Returns the statistics collector.
     *
     * @return Shared pointer to the statistics instance.
     */
    [[nodiscard]] std::shared_ptr<Statistics> statistics() const noexcept;

    /**
     * @brief Create a new channel associated with this context.
     *
     * @return A shared pointer to the newly created channel.
     */
    [[nodiscard]] std::shared_ptr<Channel> create_channel() const noexcept;

    /**
     * @brief Returns the spillable messages collection.
     *
     * @return A shared pointer to the collection.
     */
    [[nodiscard]] std::shared_ptr<SpillableMessages> spillable_messages() const noexcept;

    /**
     * @brief Create a new bounded queue associated with this context.
     *
     * @param buffer_size Maximum size of the queue.
     *
     * @return A shared pointer to the newly created bounded queue.
     */
    [[nodiscard]] std::shared_ptr<BoundedQueue> create_bounded_queue(
        std::size_t buffer_size
    ) const noexcept;

    /**
     * @brief Return a unique identifier for this context.
     *
     * The returned value uniquely identifies this Context instance. No two Context
     * objects, past or present, will ever have the same identifier within the same
     * process.
     *
     * @return A process-unique identifier for this Context.
     */
    [[nodiscard]] std::size_t uid() const noexcept;

  private:
    std::size_t const uid_;
    std::atomic<bool> is_shutdown_{false};
    std::thread::id creator_thread_id_;
    config::Options options_;
    std::shared_ptr<Communicator> comm_;
    std::shared_ptr<ProgressThread> progress_thread_;
    std::shared_ptr<CoroThreadPoolExecutor> executor_;
    std::shared_ptr<BufferResource> br_;
    std::array<std::shared_ptr<MemoryReserveOrWait>, MEMORY_TYPES.size()> memory_ = {};
    std::shared_ptr<Statistics> statistics_;
    std::shared_ptr<SpillableMessages> spillable_messages_;
    SpillManager::SpillFunctionID spill_function_id_{};
};

}  // namespace rapidsmpf::streaming
