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
#include <rapidsmpf/cuda_stream.hpp>

namespace rapidsmpf {


Buffer::Buffer(
    std::unique_ptr<std::vector<uint8_t>> host_buffer, rmm::cuda_stream_view stream
)
    : size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)},
      event_{nullptr},
      stream_{stream} {
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
    RAPIDSMPF_EXPECTS(
        device()->stream().value() == stream.value(),
        "the stream must match the device_buffer's stream",
        std::invalid_argument
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
    std::ptrdiff_t src_offset
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

    cuda_stream_join(std::array{dst.stream()}, std::array{src.stream()});
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        dst.data() + dst_offset,
        src.data() + src_offset,
        size,
        cudaMemcpyDefault,
        dst.stream()
    ));
}
}  // namespace rapidsmpf
