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
        0 <= dst_offset && dst_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(dst.size),
        "dst_offset + size can't be greater than dst.size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        0 <= src_offset && src_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(src.size),
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
