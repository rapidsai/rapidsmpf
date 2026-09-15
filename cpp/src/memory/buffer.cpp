/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <cuda/stream>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

Buffer::Buffer(
    std::unique_ptr<HostBuffer> host_buffer, cuda::stream_ref stream, MemoryType mem_type
)
    : size{host_buffer ? host_buffer->size() : 0},
      mem_type_{mem_type},
      storage_{std::move(host_buffer)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        std::get<HostBufferT>(storage_) != nullptr,
        "the host_buffer cannot be NULL",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        contains(host_buffer_types, mem_type_),
        "memory type is not suitable for a host buffer",
        std::logic_error
    );
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, MemoryType mem_type)
    : size{device_buffer ? device_buffer->size() : 0},
      mem_type_{mem_type},
      storage_{std::move(device_buffer)} {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceBufferT>(storage_) != nullptr,
        "the device buffer cannot be NULL",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        contains(device_buffer_types, mem_type_),
        "memory type is not suitable for a device buffer",
        std::logic_error
    );
    stream_ = std::get<DeviceBufferT>(storage_)->stream();
    latest_write_event_.record(stream_);
}

Buffer::Buffer(
    std::unique_ptr<disk::DiskBuffer> disk_buffer,
    std::size_t size,
    cuda::stream_ref stream
)
    : size{size},
      mem_type_{MemoryType::DISK},
      storage_{std::move(disk_buffer)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        std::get<DiskBufferT>(storage_) != nullptr,
        "the disk buffer cannot be NULL",
        std::invalid_argument
    );
}

void Buffer::throw_if_locked() const {
    RAPIDSMPF_EXPECTS(!lock_.load(std::memory_order_acquire), "the buffer is locked");
}

std::byte const* Buffer::data() const {
    throw_if_locked();
    return std::visit(
        overloaded{
            [](auto const& storage) {
                return reinterpret_cast<std::byte const*>(storage->data());
            },
            [](DiskBufferT const&) -> std::byte const* {
                RAPIDSMPF_FAIL("disk-backed buffers do not expose a data pointer");
            },
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
        overloaded{
            [](auto& storage) { return reinterpret_cast<std::byte*>(storage->data()); },
            [](DiskBufferT&) -> std::byte* {
                RAPIDSMPF_FAIL("disk-backed buffers do not expose a data pointer");
            },
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

Buffer::DeviceBufferT Buffer::release_device_buffer() {
    throw_if_locked();
    if (auto ref = std::get_if<DeviceBufferT>(&storage_)) {
        return std::move(*ref);
    }
    RAPIDSMPF_FAIL("Buffer doesn't hold a rmm::device_buffer");
}

Buffer::HostBufferT Buffer::release_host_buffer() {
    throw_if_locked();
    if (auto ref = std::get_if<HostBufferT>(&storage_)) {
        return std::move(*ref);
    }
    RAPIDSMPF_FAIL("Buffer doesn't hold a HostBuffer");
}

Buffer::DiskBufferT Buffer::release_disk_buffer() {
    throw_if_locked();
    if (auto ref = std::get_if<DiskBufferT>(&storage_)) {
        return std::move(*ref);
    }
    RAPIDSMPF_FAIL("Buffer doesn't hold a DiskBuffer");
}

void Buffer::rebind_stream(cuda::stream_ref new_stream) {
    throw_if_locked();
    if (new_stream.get() == stream_.get()) {
        return;
    }

    // Ensure the new stream does not run ahead of any work already enqueued on
    // the current stream.
    latest_write_event_.stream_wait(new_stream);
    stream_ = new_stream;

    std::visit(
        [&](auto&& storage) {
            if constexpr (!std::is_same_v<std::decay_t<decltype(storage)>, DiskBufferT>) {
                storage->set_stream(new_stream);
            }
        },
        storage_
    );
}

namespace {

Duration copy_from_disk(
    Buffer& destination,
    disk::DiskBuffer const& source,
    std::size_t size,
    std::ptrdiff_t destination_offset,
    std::ptrdiff_t source_offset
) {
    RAPIDSMPF_EXPECTS(
        source.disk_resource() != nullptr, "DiskBuffer has no DiskResource"
    );

    auto const start = Clock::now();
    destination.write_access([&](std::byte* destination_data, cuda::stream_ref) {
        auto const transferred = source.disk_resource()->read(
            source.path(),
            destination_data + destination_offset,
            size,
            destination.mem_type(),
            static_cast<std::size_t>(source_offset)
        );
        RAPIDSMPF_EXPECTS(
            transferred == size,
            "disk read transferred " + format_nbytes(transferred) + " of "
                + format_nbytes(size),
            std::runtime_error
        );
    });
    destination.stream().sync();
    return Clock::now() - start;
}

Duration copy_to_disk(
    disk::DiskBuffer const& destination,
    Buffer const& source,
    std::size_t size,
    std::ptrdiff_t destination_offset,
    std::ptrdiff_t source_offset
) {
    RAPIDSMPF_EXPECTS(
        destination.disk_resource() != nullptr, "DiskBuffer has no DiskResource"
    );

    auto const start = Clock::now();
    auto const transferred = destination.disk_resource()->write(
        destination.path(),
        source.data() + source_offset,
        size,
        source.mem_type(),
        static_cast<std::size_t>(destination_offset)
    );
    RAPIDSMPF_EXPECTS(
        transferred == size,
        "disk write transferred " + format_nbytes(transferred) + " of "
            + format_nbytes(size),
        std::runtime_error
    );
    return Clock::now() - start;
}

}  // namespace

void buffer_copy(
    std::shared_ptr<Statistics> statistics,
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
        return;
    }
    RAPIDSMPF_EXPECTS(statistics != nullptr, "the statistics pointer cannot be NULL");

    auto const src_is_disk = src.mem_type() == MemoryType::DISK;
    auto const dst_is_disk = dst.mem_type() == MemoryType::DISK;

    if (src_is_disk && dst_is_disk) {
        RAPIDSMPF_FAIL("disk-to-disk copy is not supported", std::invalid_argument);
    }
    if (dst_is_disk) {
        src.latest_write_event().host_wait();
        auto const duration = copy_to_disk(
            *dst.get_storage<Buffer::DiskBufferT>(), src, size, dst_offset, src_offset
        );
        statistics->record_copy(src.mem_type(), dst.mem_type(), size, duration);
        return;
    }
    if (src_is_disk) {
        dst.latest_write_event().host_wait();
        auto const duration = copy_from_disk(
            dst, *src.get_storage<Buffer::DiskBufferT>(), size, dst_offset, src_offset
        );
        statistics->record_copy(src.mem_type(), dst.mem_type(), size, duration);
        return;
    }

    src.latest_write_event().stream_wait(dst.stream());
    StreamOrderedTiming timing{dst.stream(), statistics};
    dst.write_access([&](std::byte* dst_data, cuda::stream_ref stream) {
        RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(
            dst_data + dst_offset, src.data() + src_offset, size, stream
        ));
    });
    // after the dst.write_access(), its last_write_event is recorded on dst.stream(). So,
    // we need the src.stream() to wait for that event.
    dst.latest_write_event().stream_wait(src.stream());
    statistics->record_copy(src.mem_type(), dst.mem_type(), size, std::move(timing));
}

}  // namespace rapidsmpf
