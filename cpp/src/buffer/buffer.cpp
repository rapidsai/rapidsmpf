/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>

#include <cuda_runtime.h>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>

namespace rapidsmpf {

namespace {
// Check that `ptr` isn't null.
template <typename T>
[[nodiscard]] std::unique_ptr<T> check_null(std::unique_ptr<T> ptr) {
    RAPIDSMPF_EXPECTS(ptr, "unique pointer cannot be null", std::invalid_argument);
    return ptr;
}
}  // namespace

/**
 * @brief CUDA event to provide synchronization among set of chunks.
 *
 * This event is used to serve as a synchronization point for a set of chunks
 * given a user-specified stream.
 */
class Event {
  public:
    /**
     * @brief Construct a CUDA event for a given stream.
     *
     * @param stream CUDA stream used for device memory operations
     * @param log Logger to warn if object is destroyed before event is ready.
     */
    Event(rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream));
    }

    /**
     * @brief Destructor for Event.
     *
     * Cleans up the CUDA event if one was created. If the event is not done,
     * it will log a warning.
     */
    ~Event() {
        // Mark as destroying - if we fail, another thread is already destroying
        bool expected = false;
        if (!destroying_.compare_exchange_strong(expected, true)) {
            return;
        }

        // Finally acquire the mutex and destroy the event
        std::lock_guard<std::mutex> lock(mutex_);
        cudaEventDestroy(event_);
    }

    /**
     * @brief Check if the CUDA event has been completed.
     *
     * @return true if the event has been completed, false otherwise.
     */
    [[nodiscard]] bool is_ready() {
        // Fast path: if done or being destroyed, return immediately
        if (done_.load(std::memory_order_relaxed)
            || destroying_.load(std::memory_order_acquire))
        {
            return true;
        }

        // Acquire mutex and check destroying_ again, if being destroyed, return the
        // previous value of done_.
        std::lock_guard<std::mutex> lock(mutex_);
        if (destroying_.load(std::memory_order_acquire)) {
            return done_.load(std::memory_order_relaxed);
        }

        // If we're not destroying, check if the event is ready
        return cudaEventQuery(event_) == cudaSuccess;
    }

  private:
    cudaEvent_t event_;  ///< CUDA event used to track device memory allocation
    // Communicator::Logger&
    //     log_;  ///< Logger to warn if object is destroyed before event is ready
    std::atomic<bool> done_{false
    };  ///< Cache of the event status to avoid unnecessary queries.
    mutable std::mutex mutex_;  ///< Protects access to event_
    std::atomic<bool> destroying_{false};  ///< Flag to indicate destruction in progress
};

Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br)
    : br{br},
      size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)},
      event_{nullptr} {
    RAPIDSMPF_EXPECTS(
        std::get<HostStorageT>(storage_) != nullptr, "the host_buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(
    std::unique_ptr<rmm::device_buffer> device_buffer,
    rmm::cuda_stream_view stream,
    BufferResource* br
)
    : br{br},
      size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)},
      event_{std::make_unique<Event>(stream)} {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceStorageT>(storage_) != nullptr, "the device buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

void* Buffer::data() {
    return std::visit([](auto&& storage) -> void* { return storage->data(); }, storage_);
}

void const* Buffer::data() const {
    return std::visit([](auto&& storage) -> void* { return storage->data(); }, storage_);
}

std::unique_ptr<Buffer> Buffer::copy(rmm::cuda_stream_view stream) const {
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) -> std::unique_ptr<Buffer> {
                return std::unique_ptr<Buffer>(
                    new Buffer{std::make_unique<std::vector<uint8_t>>(*storage), br}
                );
            },
            [&](const DeviceStorageT& storage) -> std::unique_ptr<Buffer> {
                auto new_buffer = std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        storage->data(), storage->size(), stream, br->device_mr()
                    ),
                    stream,
                    br
                });
                return new_buffer;
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy(MemoryType target, rmm::cuda_stream_view stream)
    const {
    if (mem_type() == target) {
        return copy(stream);
    }

    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) -> std::unique_ptr<Buffer> {
                auto new_buffer = std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        storage->data(), storage->size(), stream, br->device_mr()
                    ),
                    stream,
                    br
                });
                return new_buffer;
            },
            [&](const DeviceStorageT& storage) -> std::unique_ptr<Buffer> {
                auto ret = std::make_unique<std::vector<uint8_t>>(storage->size());
                RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                    ret->data(),
                    storage->data(),
                    storage->size(),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                auto new_buffer = std::unique_ptr<Buffer>(new Buffer{std::move(ret), br});

                // The event is created here instead of the constructor because the
                // memcpy is async, but the buffer is created on the host.
                new_buffer->event_ = std::make_unique<Event>(stream);

                return new_buffer;
            }
        },
        storage_
    );
}

bool Buffer::is_ready() const {
    if (event_ == nullptr) {
        return true;  // No device memory operation was performed
    }
    return event_->is_ready();
}

}  // namespace rapidsmpf
