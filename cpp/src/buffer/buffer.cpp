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

Buffer::Event::Event(rmm::cuda_stream_view stream) {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream));
}

Buffer::Event::~Event() {
    // First mark as destroying to prevent new readers
    destroying_.store(true, std::memory_order_release);

    // Wait for any pending is_ready() calls to complete
    while (active_readers_.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }

    // Now safe to destroy as no readers can start and no readers are active
    std::lock_guard<std::mutex> lock(mutex_);
    // TODO: if we're being destroyed, warn the user if the event is
    // not completed.
    cudaEventDestroy(event_);
}

[[nodiscard]] bool Buffer::Event::is_ready() {
    // First register as an active reader
    active_readers_.fetch_add(1, std::memory_order_acquire);

    auto exit_if_destroying = [&]() -> std::optional<bool> {
        if (destroying_.load(std::memory_order_acquire)) {
            bool done = done_.load(std::memory_order_relaxed);
            active_readers_.fetch_sub(1, std::memory_order_release);
            return done;
        }
        return std::nullopt;
    };

    // Early exit if destroying
    if (auto destroying = exit_if_destroying()) {
        return *destroying;
    }

    // Check if we've already determined it's done
    if (done_.load(std::memory_order_relaxed)) {
        active_readers_.fetch_sub(1, std::memory_order_release);
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // Recheck destroying_ under lock
    if (auto destroying = exit_if_destroying()) {
        return *destroying;
    }

    bool done = done_.load(std::memory_order_relaxed);
    if (!done) {
        bool is_complete = (cudaEventQuery(event_) == cudaSuccess);
        done_.store(is_complete, std::memory_order_release);
        done = is_complete;
    }

    active_readers_.fetch_sub(1, std::memory_order_release);
    return done;
}

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
    BufferResource* br,
    std::shared_ptr<Event> event
)
    : br{br},
      size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)},
      event_{event ? event : std::make_shared<Event>(stream)} {
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
