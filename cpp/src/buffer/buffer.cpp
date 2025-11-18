/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <cuda/std/cstdint>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/host_buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/cuda_stream.hpp>

namespace rapidsmpf {


Buffer::Buffer(std::unique_ptr<HostBuffer> host_buffer, rmm::cuda_stream_view stream)
    : size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)},
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
    latest_write_event_.record(stream_);
}

void Buffer::throw_if_locked() const {
    RAPIDSMPF_EXPECTS(!lock_.load(std::memory_order_acquire), "the buffer is locked");
}

Buffer::HostStorageT const& Buffer::host() const {
    throw_if_locked();
    if (const auto* ref = std::get_if<HostStorageT>(&storage_)) {
        return *ref;
    } else {
        RAPIDSMPF_FAIL("Buffer is not host memory");
    }
}

Buffer::HostStorageT& Buffer::host() {
    throw_if_locked();
    if (auto ref = std::get_if<HostStorageT>(&storage_)) {
        return *ref;
    } else {
        RAPIDSMPF_FAIL("Buffer is not host memory");
    }
}

Buffer::DeviceStorageT& Buffer::device() {
    throw_if_locked();
    if (auto ref = std::get_if<DeviceStorageT>(&storage_)) {
        return *ref;
    } else {
        RAPIDSMPF_FAIL("Buffer is not device memory");
    }
}

Buffer::DeviceStorageT const& Buffer::device() const {
    throw_if_locked();
    if (const auto* ref = std::get_if<DeviceStorageT>(&storage_)) {
        return *ref;
    } else {
        RAPIDSMPF_FAIL("Buffer is not device memory");
    }
}

std::byte const* Buffer::data() const {
    throw_if_locked();
    return std::visit(
        [](auto&& storage) -> std::byte const* {
            return reinterpret_cast<std::byte const*>(storage->data());
        },
        storage_
    );
}

std::byte* Buffer::exclusive_data_access() {
    RAPIDSMPF_EXPECTS(is_latest_write_done(), "the latest write isn't done");

    bool expected = false;
    RAPIDSMPF_EXPECTS(
        lock_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire
        ),
        "the buffer is already locked"
    );
    return std::visit(
        [](auto&& storage) -> std::byte* {
            return reinterpret_cast<std::byte*>(storage->data());
        },
        storage_
    );
}

void Buffer::unlock() {
    lock_.store(false, std::memory_order_release);
}

bool Buffer::is_latest_write_done() const {
    throw_if_locked();
    return size == 0 || latest_write_event_.is_ready();
}

Buffer::DeviceStorageT Buffer::release_device() {
    throw_if_locked();
    return std::move(device());
}

Buffer::HostStorageT Buffer::release_host() {
    throw_if_locked();
    return std::move(host());
}

void buffer_copy(
    Buffer& dst,
    Buffer const& src,
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

    // We have to sync both before *and* after the memcpy. Otherwise, `src.stream()`
    // might deallocate `src` before the memcpy enqueued on `dst.stream()` has completed.
    // This is too much synchronization. The source source buffer has an event. That's all
    // we need to wait for.
    cuda_stream_join(dst.stream(), src.stream());
    dst.write_access([&](std::byte* dst_data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            dst_data + dst_offset,
            src.data() + src_offset,
            size,
            cudaMemcpyDefault,
            stream
        ));
    });
    // Again, this is too much. We only need to wait on the event in the destination
    // buffer. We don't need to allocate an event.
    cuda_stream_join(src.stream(), dst.stream());
}

}  // namespace rapidsmpf
