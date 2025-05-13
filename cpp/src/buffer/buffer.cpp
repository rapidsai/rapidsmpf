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
    // TODO: if we're being destroyed, warn the user if the event is
    // not completed.
    cudaEventDestroy(event_);
}

[[nodiscard]] bool Buffer::Event::is_ready() {
    if (!done_.load(std::memory_order_relaxed)) {
        bool result = (cudaEventQuery(event_) == cudaSuccess);
        done_.store(result, std::memory_order_relaxed);
        return result;
    }

    return true;
}

void Buffer::Event::wait() {
    if (!done_.load(std::memory_order_relaxed)) {
        RAPIDSMPF_CUDA_TRY(cudaEventSynchronize(event_));
        done_.store(true, std::memory_order_relaxed);
    }
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
      // Use the provided event if it exists, otherwise create a new event to track the
      // async copy only if the buffer is not empty
      event_{
          event      ? event
          : size > 0 ? std::make_shared<Event>(stream)
                     : nullptr
      } {
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
                new_buffer->event_ = std::make_shared<Event>(stream);

                return new_buffer;
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    std::ptrdiff_t offset, std::ptrdiff_t length, rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(offset <= std::ptrdiff_t(size), "offset can't be more than size");
    RAPIDSMPF_EXPECTS(
        offset + length <= std::ptrdiff_t(size), "offset + length can't be more than size"
    );
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {  // host -> host
                // wait for this buffer to be ready, as host-to-host copy is synchronous
                wait_for_ready();
                auto host_buf = std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<std::vector<uint8_t>>(
                        storage->begin() + offset, storage->begin() + offset + length
                    ),
                    br
                });
                host_buf->override_event(event_);  // if there was an event, use it
                return host_buf;
            },
            [&](DeviceStorageT const& storage) {  // device -> device
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        static_cast<cuda::std::byte*>(storage->data()) + offset,
                        length,
                        stream,
                        br->device_mr()
                    ),
                    stream,
                    br
                });
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    MemoryType target,
    std::ptrdiff_t offset,
    std::ptrdiff_t length,
    rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(offset <= std::ptrdiff_t(size), "offset can't be more than size");
    RAPIDSMPF_EXPECTS(
        offset + length <= std::ptrdiff_t(size), "offset + length can't be more than size"
    );

    if (mem_type() == target) {
        return copy_slice(offset, length, stream);
    }

    // Implement the copy between each possible memory types (both directions).
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {  // host -> device
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        static_cast<uint8_t const*>(storage->data()) + offset,
                        length,
                        stream,
                        br->device_mr()
                    ),
                    stream,
                    br
                });
            },
            [&](DeviceStorageT const& storage) {  // device -> host
                {
                    auto ret = std::make_unique<std::vector<uint8_t>>(length);
                    RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                        ret->data(),
                        static_cast<cuda::std::byte const*>(storage->data()) + offset,
                        size_t(length),
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                    auto host_buf =
                        std::unique_ptr<Buffer>(new Buffer{std::move(ret), br});
                    // Create a new event to track the async copy only if length > 0
                    if (length > 0) {
                        host_buf->override_event(std::make_shared<Event>(stream));
                    }
                    return host_buf;
                }
            }
        },
        storage_
    );
}

std::ptrdiff_t Buffer::copy_to(
    Buffer& dest, std::ptrdiff_t dest_offset, rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(
        dest.size - size_t(dest_offset) >= size,
        "destination buffer is too small",
        std::invalid_argument
    );

    if (size == 0) {  // empty buffer, nothing to do
        return 0;
    }

    cudaMemcpyKind kind;

    // if both buffers are on the host, use memcpy, otherwise, use cudaMemcpyAsync
    if (mem_type() == MemoryType::HOST && dest.mem_type() == MemoryType::HOST) {
        // if the source buffer is not ready, return 0
        if (!is_ready()) {
            return 0;
        }
        // src buffer is ready, copy the data
        std::memcpy(static_cast<uint8_t*>(dest.data()) + dest_offset, data(), size);
        return std::ptrdiff_t(size);
    } else if (mem_type() == MemoryType::HOST) {
        kind = cudaMemcpyHostToDevice;
    } else if (dest.mem_type() == MemoryType::HOST) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }

    RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
        static_cast<cuda::std::byte*>(dest.data()) + dest_offset,
        data(),
        size,
        kind,
        stream
    ));
    return std::ptrdiff_t(size);
}

bool Buffer::is_ready() const {
    return !event_ || event_->is_ready();
}

void Buffer::wait_for_ready() const {
    if (event_) {
        event_->wait();
    }
}

}  // namespace rapidsmpf
