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

Buffer::Buffer(std::unique_ptr<DiskBuffer> disk_buffer, cuda::stream_ref stream)
    : size{disk_buffer ? disk_buffer->size() : 0},
      mem_type_{MemoryType::DISK},
      storage_{std::move(disk_buffer)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        std::get<DiskBufferT>(storage_) != nullptr,
        "the disk buffer cannot be NULL",
        std::invalid_argument
    );
    latest_write_event_.record(stream_);
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

    std::visit([&](auto&& storage) { storage->set_stream(new_stream); }, storage_);
}

void Buffer::record_write_event() {
    throw_if_locked();
    latest_write_event_.record(stream_);
}

namespace {

Duration copy_from_disk(
    Buffer& dst,
    Buffer const& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset
) {
    auto const& disk_src = *src.get_storage<Buffer::DiskBufferT>();
    auto const file_size = disk_src.file_size();
    auto const offset = static_cast<std::size_t>(src_offset);
    RAPIDSMPF_EXPECTS(
        offset <= file_size && size <= file_size - offset,
        "src_offset + size can't be greater than the backing file size",
        std::invalid_argument
    );

    auto const start = Clock::now();
    dst.write_access([&](std::byte* dst_data, cuda::stream_ref stream) {
        auto const transferred = disk_src.disk_resource()->read(
            disk_src.path(),
            dst_data + dst_offset,
            size,
            dst.mem_type(),
            stream,
            src_offset
        );
        RAPIDSMPF_EXPECTS(
            transferred == size,
            "disk read transferred " + format_nbytes(transferred) + " of "
                + format_nbytes(size),
            std::runtime_error
        );
    });
    return Clock::now() - start;
}

Duration copy_to_disk(
    Buffer& dst,
    Buffer const& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset
) {
    auto const& disk_dst = *dst.get_storage<Buffer::DiskBufferT>();

    auto const start = Clock::now();
    auto const transferred = disk_dst.disk_resource()->write(
        disk_dst.path(),
        src.data() + src_offset,
        size,
        src.mem_type(),
        dst.stream(),
        dst_offset
    );
    RAPIDSMPF_EXPECTS(
        transferred == size,
        "disk write transferred " + format_nbytes(transferred) + " of "
            + format_nbytes(size),
        std::runtime_error
    );
    dst.record_write_event();
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

    // TODO: currently disk read/write is synchronous. Use stream ordered timing, once
    // disk resource supports async read/write.
    std::variant<StreamOrderedTiming, Duration> timing;
    src.latest_write_event().stream_wait(dst.stream());
    if (dst_is_disk) {
        timing = copy_to_disk(dst, src, size, dst_offset, src_offset);
    } else if (src_is_disk) {
        timing = copy_from_disk(dst, src, size, dst_offset, src_offset);
    } else {
        timing = StreamOrderedTiming(dst.stream(), statistics);
        dst.write_access([&](std::byte* dst_data, cuda::stream_ref stream) {
            RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(
                dst_data + dst_offset, src.data() + src_offset, size, stream
            ));
        });
    }
    // after the dst.write_access(), its last_write_event is recorded on dst.stream(). So,
    // we need the src.stream() to wait for that event.
    dst.latest_write_event().stream_wait(src.stream());
    std::visit(
        [&](auto&& timing) {
            statistics->record_copy(
                src.mem_type(), dst.mem_type(), size, std::move(timing)
            );
        },
        timing
    );
}

}  // namespace rapidsmpf
