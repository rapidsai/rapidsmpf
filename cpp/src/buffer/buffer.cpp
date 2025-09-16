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


Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer)
    : size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)},
      event_{nullptr} {
    RAPIDSMPF_EXPECTS(
        std::get<HostStorageT>(storage_) != nullptr, "the host_buffer cannot be NULL"
    );
}

Buffer::Buffer(
    std::unique_ptr<rmm::device_buffer> device_buffer,
    rmm::cuda_stream_view stream,
    std::shared_ptr<CudaEvent> event
)
    : size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)},
      // Use the provided event if it exists, otherwise create a new event to track the
      // async copy only if the buffer is not empty
      event_{
          event      ? event
          : size > 0 ? CudaEvent::make_shared_record(stream)
                     : nullptr
      } {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceStorageT>(storage_) != nullptr, "the device buffer cannot be NULL"
    );
}

std::byte* Buffer::data() {
    return std::visit(
        [](auto&& storage) -> std::byte* {
            return reinterpret_cast<std::byte*>(storage->data());
        },
        storage_
    );
}

std::byte const* Buffer::data() const {
    return std::visit(
        [](auto&& storage) -> std::byte const* {
            return reinterpret_cast<std::byte const*>(storage->data());
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
        target_reserv.size() >= length, "reservation is too small", std::overflow_error
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
        auto out_buf = target_reserv.br()->allocate(length, stream, target_reserv);

        if (length > 0) {
            // if this buffer has an event, ask the current stream to wait for it, before
            // performing the memcpy
            if (event_) {
                event_->stream_wait(stream);
            }

            RAPIDSMPF_CUDA_TRY(
                cudaMemcpyAsync(out_buf->data(), source + offset, length, kind, stream)
            );
            // override the event to track the async copy, if the copy is not host-to-host
            // cudaMemcpyAsync is synchronous for host-to-host copies
            // ref: https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
            if (kind != cudaMemcpyHostToHost) {
                out_buf->override_event(CudaEvent::make_shared_record(stream));
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

bool Buffer::is_ready() const {
    return !event_ || event_->is_ready();
}

void Buffer::wait_for_ready() const {
    if (event_) {
        event_->host_wait();
    }
}

void buffer_copy(
    Buffer& dst,
    Buffer& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset,
    rmm::cuda_stream_view stream,
    bool attach_cuda_event
) {
    RAPIDSMPF_EXPECTS(
        &dst != &src,
        "the source and destination cannot be the same buffer",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        dst_offset >= 0 && std::cmp_less_equal(dst_offset, dst.size),
        "dst_offset can't be greater than dst.size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        dst_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(dst.size),
        "dst_offset + size can't be greater than dst.size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        src_offset >= 0 && std::cmp_less_equal(src_offset, src.size),
        "src_offset can't be greater than src.size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        src_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(src.size),
        "src_offset + size can't be greater than src.size",
        std::invalid_argument
    );
    if (size == 0) {
        return;  // Nothing to copy.
    }

    // Make sure we wait on any buffer event.
    if (auto e = dst.get_event()) {
        e->stream_wait(stream);
    }
    if (auto e = src.get_event()) {
        e->stream_wait(stream);
    }

    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        dst.data() + dst_offset, src.data() + src_offset, size, cudaMemcpyDefault, stream
    ));

    // Override the event to track the async copy.
    if (attach_cuda_event) {
        dst.override_event(CudaEvent::make_shared_record(stream));
    }
}
}  // namespace rapidsmpf
