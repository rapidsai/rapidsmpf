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

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer)
    : size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)} {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceStorageT>(storage_) != nullptr,
        "the device buffer cannot be NULL",
        std::invalid_argument
    );
    stream_ = std::get<DeviceStorageT>(storage_)->stream();
    // Create a new event to track the async copy only if the buffer is not empty.
    if (size > 0) {
        event_ = CudaEvent::make_shared_record(stream_);
    }
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

void buffer_copy(
    Buffer& dst,
    Buffer& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset,
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

    // We have to sync both before *and* after the memcpy. Otherwise, `src.stream()`
    // might deallocate `src` before the memcpy enqueued on `dst.stream()` has completed.
    cuda_stream_join(std::array{dst.stream()}, std::array{src.stream()});
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        dst.data() + dst_offset,
        src.data() + src_offset,
        size,
        cudaMemcpyDefault,
        dst.stream()
    ));
    cuda_stream_join(std::array{src.stream()}, std::array{dst.stream()});

    // Override the event to track the async copy.
    if (attach_cuda_event) {
        src.override_event(CudaEvent::make_shared_record(src.stream()));
        dst.override_event(CudaEvent::make_shared_record(dst.stream()));
    }
}
}  // namespace rapidsmpf
