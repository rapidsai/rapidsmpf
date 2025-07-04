/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <cuda/std/cstdint>

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
    if (!done_.load(std::memory_order_acquire)) {
        auto result = cudaEventQuery(event_);
        done_.store(result == cudaSuccess, std::memory_order_release);
        if (result != cudaSuccess && result != cudaErrorNotReady) {
            RAPIDSMPF_CUDA_TRY(result);
        }
        return result == cudaSuccess;
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
    std::ptrdiff_t offset,
    std::size_t length,
    MemoryReservation& target_reserv,
    rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(
        target_reserv.size() >= length, "reservation is too small", std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        offset >= 0 && std::cmp_less_equal(offset, size),
        "offset can't be greater than size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        offset + std::ptrdiff_t(length) <= std::ptrdiff_t(size),
        "offset + length can't be greater than size",
        std::invalid_argument
    );

    // lambda to allocate a buffer and perform a memcpy
    auto do_alloc_and_cuda_memcpy_async = [&](cudaMemcpyKind kind,
                                              cuda::std::uint8_t* source) {
        // allocate buffer using the target reservation
        auto out_buf = target_reserv.br()->allocate(
            target_reserv.mem_type(), length, stream, target_reserv
        );

        if (length > 0) {
            // if this buffer has an event, ask the current stream to wait for it, before
            // performing the memcpy
            if (event_) {
                RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(stream, event_->event()));
            }

            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                static_cast<cuda::std::uint8_t*>(out_buf->data()),
                source + offset,
                length,
                kind,
                stream
            ));
            // override the event to track the async copy, if the copy is not host-to-host
            // cudaMemcpyAsync is synchronous for host-to-host copies
            // ref: https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
            if (kind != cudaMemcpyHostToHost) {
                out_buf->override_event(std::make_shared<Buffer::Event>(stream));
            }
        }

        return out_buf;
    };

    // Implement the copy between each possible memory types (both directions).
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {
                switch (target_reserv.mem_type()) {
                case MemoryType::HOST:  // host -> host
                    return do_alloc_and_cuda_memcpy_async(
                        cudaMemcpyHostToHost, storage->data()
                    );
                case MemoryType::DEVICE:  // host -> device
                    return do_alloc_and_cuda_memcpy_async(
                        cudaMemcpyHostToDevice, storage->data()
                    );
                }
                RAPIDSMPF_FAIL("Invalid memory type");  // unreachable
            },
            [&](DeviceStorageT const& storage) {
                switch (target_reserv.mem_type()) {
                case MemoryType::HOST:  // device -> host
                    return do_alloc_and_cuda_memcpy_async(
                        cudaMemcpyDeviceToHost,
                        static_cast<cuda::std::uint8_t*>(storage->data())
                    );
                case MemoryType::DEVICE:  // device -> device
                    return do_alloc_and_cuda_memcpy_async(
                        cudaMemcpyDeviceToDevice,
                        static_cast<cuda::std::uint8_t*>(storage->data())
                    );
                }
                RAPIDSMPF_FAIL("Invalid memory type");  // unreachable
            }
        },
        storage_
    );
}

std::ptrdiff_t Buffer::copy_to(
    Buffer& dest,
    std::ptrdiff_t dest_offset,
    rmm::cuda_stream_view stream,
    bool attach_event
) const {
    RAPIDSMPF_EXPECTS(
        dest_offset >= 0 && std::cmp_less_equal(dest_offset, dest.size),
        "destination offset can't be greater than destination buffer size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        dest.size - size_t(dest_offset) >= size,
        "destination buffer is too small",
        std::invalid_argument
    );

    if (size == 0) {  // empty buffer, nothing to do
        return 0;
    }

    // lambda to perform a memcpy and optionally attach an event
    auto do_cuda_memcpy_async = [&](cudaMemcpyKind kind, cuda::std::uint8_t* source) {
        // if this buffer has an event, ask the current stream to wait for it, before
        // performing the memcpy
        if (event_) {
            RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(stream, event_->event()));
        }

        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            static_cast<cuda::std::uint8_t*>(dest.data()) + dest_offset,
            source,
            size,
            kind,
            stream
        ));

        // if the copy is not host-to-host, override the event to track the async copy
        if (attach_event && kind != cudaMemcpyHostToHost) {
            dest.override_event(std::make_shared<Buffer::Event>(stream));
        }

        return std::ptrdiff_t(size);
    };

    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {
                switch (dest.mem_type()) {
                case MemoryType::HOST:  // host -> host
                    return do_cuda_memcpy_async(cudaMemcpyHostToHost, storage->data());
                case MemoryType::DEVICE:  // host -> device
                    return do_cuda_memcpy_async(cudaMemcpyHostToDevice, storage->data());
                }
                RAPIDSMPF_FAIL("Invalid memory type");  // unreachable
            },
            [&](const DeviceStorageT& storage) {
                switch (dest.mem_type()) {
                case MemoryType::HOST:  // device -> host
                    return do_cuda_memcpy_async(
                        cudaMemcpyDeviceToHost,
                        static_cast<cuda::std::uint8_t*>(storage->data())
                    );
                case MemoryType::DEVICE:  // device -> device
                    return do_cuda_memcpy_async(
                        cudaMemcpyDeviceToDevice,
                        static_cast<cuda::std::uint8_t*>(storage->data())
                    );
                }
                RAPIDSMPF_FAIL("Invalid memory type");  // unreachable
            }
        },
        storage_
    );
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
