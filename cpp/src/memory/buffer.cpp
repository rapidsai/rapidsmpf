/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cuda/std/cstdint>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

namespace rapidsmpf {


Buffer::Buffer(
    std::unique_ptr<HostBuffer> host_buffer,
    rmm::cuda_stream_view stream,
    MemoryType mem_type
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
    std::unique_ptr<FixedSizedHostBuffer> fixed_host_buffer,
    rmm::cuda_stream_view stream,
    MemoryType mem_type
)
    : size{fixed_host_buffer ? fixed_host_buffer->total_size() : 0},
      mem_type_{mem_type},
      storage_{std::move(fixed_host_buffer)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        std::get<FixedSizedHostBufferT>(storage_) != nullptr,
        "the fixed_host_buffer cannot be NULL",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        contains(pinned_buffer_types, mem_type_),
        "memory type is not suitable for a pinned buffer",
        std::logic_error
    );
}

void Buffer::throw_if_locked() const {
    RAPIDSMPF_EXPECTS(!lock_.load(std::memory_order_acquire), "the buffer is locked");
}

std::byte const* Buffer::data() const {
    throw_if_locked();
    return std::visit(
        overloaded{
            [](FixedSizedHostBufferT const&) -> std::byte const* {
                RAPIDSMPF_FAIL("data() is not supported for FixedSizedHostBuffer");
            },
            [](auto const& storage) -> std::byte const* {
                return reinterpret_cast<std::byte const*>(storage->data());
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
            [](FixedSizedHostBufferT&) -> std::byte* {
                RAPIDSMPF_FAIL(
                    "exclusive_data_access() is not supported for FixedSizedHostBuffer"
                );
            },
            [](auto& storage) -> std::byte* {
                return reinterpret_cast<std::byte*>(storage->data());
            },
        },
        storage_
    );
}

std::vector<std::byte*> Buffer::exclusive_data_access_blocks() {
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
            [](FixedSizedHostBufferT& buf) -> std::vector<std::byte*> {
                auto blocks = buf->blocks();
                return {blocks.begin(), blocks.end()};
            },
            [](auto& storage) -> std::vector<std::byte*> {
                return {reinterpret_cast<std::byte*>(storage->data())};
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

Buffer::FixedSizedHostBufferT Buffer::release_fixed_sized_host_buffer() {
    throw_if_locked();
    if (auto ref = std::get_if<FixedSizedHostBufferT>(&storage_)) {
        return std::move(*ref);
    }
    RAPIDSMPF_FAIL("Buffer doesn't hold a FixedSizedHostBuffer");
}

void Buffer::rebind_stream(rmm::cuda_stream_view new_stream) {
    throw_if_locked();
    if (new_stream.value() == stream_.value()) {
        return;
    }

    // Ensure the new stream does not run ahead of any work already enqueued on
    // the current stream.
    latest_write_event_.stream_wait(new_stream);
    stream_ = new_stream;

    std::visit([&](auto& storage) { storage->set_stream(new_stream); }, storage_);
}

namespace {

void cuda_memcpy_batch_async(
    std::span<void const*> const src_ptrs,
    std::span<void const*> const dst_ptrs,
    std::span<std::size_t> const sizes,
    rmm::cuda_stream_view stream
) {
    RAPIDSMPF_EXPECTS(
        src_ptrs.size() == dst_ptrs.size() && src_ptrs.size() == sizes.size(),
        "the number of source and destination pointers must be the same",
        std::invalid_argument
    );

    // cudaMemcpyBatchAsync does not support the null/legacy stream or the per-thread
    // default stream — passing either returns cudaErrorInvalidValue. Fall back to
    // individual cudaMemcpyAsync calls in that case.
    if (stream.value() == nullptr) {
        for (std::size_t i = 0; i < src_ptrs.size(); ++i) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                const_cast<void*>(dst_ptrs[i]), src_ptrs[i], sizes[i], cudaMemcpyDefault, stream.value()
            ));
        }
        return;
    }

    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::array<size_t, 1> attrsIdxs{0};

#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(13000)
    RAPIDSMPF_CUDA_TRY(cudaMemcpyBatchAsync(
        dst_ptrs.data(),
        src_ptrs.data(),
        sizes.data(),
        src_ptrs.size(),
        &attrs,
        attrsIdxs.data(),
        attrsIdxs.size(),
        stream.value()
    ));
#else
    size_t failIdx{};
    RAPIDSMPF_CUDA_TRY(cudaMemcpyBatchAsync(
        const_cast<void**>(dst_ptrs.data()),
        const_cast<void**>(src_ptrs.data()),
        sizes.data(),
        src_ptrs.size(),
        &attrs,
        attrsIdxs.data(),
        attrsIdxs.size(),
        &failIdx,
        stream.value()
    ));
#endif
}

}  // namespace

void Buffer::copy_to(
    Buffer& dst, std::size_t size, std::ptrdiff_t dst_offset, std::ptrdiff_t src_offset
) const {
    RAPIDSMPF_EXPECTS(
        &dst != this,
        "the source and destination cannot be the same buffer",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        0 <= dst_offset && dst_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(dst.size),
        "dst_offset + size can't be greater than dst.size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        0 <= src_offset
            && src_offset + std::ptrdiff_t(size) <= std::ptrdiff_t(this->size),
        "src_offset + size can't be greater than src.size",
        std::invalid_argument
    );
    if (size == 0) {
        return;
    }

    auto block_bounds = [](Buffer const& buf, size_t offset) -> std::span<std::byte> {
        return std::visit(
            overloaded{
                [&](FixedSizedHostBufferT const& buf) {
                    auto block_idx = offset / buf->block_size();
                    auto block_offset = offset % buf->block_size();
                    return buf->block_data(block_idx).subspan(block_offset);
                },
                [&](auto& buf) {
                    return std::span<std::byte>(
                        reinterpret_cast<std::byte*>(buf->data()) + offset,
                        buf->size() - offset
                    );
                },
            },
            buf.storage_
        );
    };

    auto n_byte_boundaries = [](Buffer const& buf, size_t offset, size_t size) -> size_t {
        return std::visit(
            overloaded{
                [&](FixedSizedHostBufferT const& buf) -> size_t {
                    const size_t block_sz = buf->block_size();
                    const size_t first_block = offset / block_sz;
                    const size_t last_block = (offset + size - 1) / block_sz;
                    return 1 + last_block - first_block;
                },
                [&]([[maybe_unused]] auto& buf) -> size_t { return 1; },
            },
            buf.storage_
        );
    };

    latest_write_event().stream_wait(dst.stream());


    std::vector<void const*> src_ptrs;
    std::vector<void const*> dst_ptrs;
    std::vector<std::size_t> sizes;

    // use a heuristic to reserve the vectors
    size_t approx_num_parts =
        n_byte_boundaries(*this, static_cast<size_t>(src_offset), size)
        + n_byte_boundaries(dst, static_cast<size_t>(dst_offset), size);
    src_ptrs.reserve(approx_num_parts);
    dst_ptrs.reserve(approx_num_parts);
    sizes.reserve(approx_num_parts);

    size_t offset = 0;

    // Prime the running block state for both buffers — one std::visit each.
    auto src_span = block_bounds(*this, static_cast<size_t>(src_offset));
    auto dst_span = block_bounds(dst, static_cast<size_t>(dst_offset));
    std::byte* src_ptr = src_span.data();
    std::byte* dst_ptr = dst_span.data();
    size_t src_rem = src_span.size();
    size_t dst_rem = dst_span.size();

    // Walk block boundaries for src and dst independently: block_bounds is only
    // called again when a buffer actually crosses a block boundary, rather than
    // on every loop iteration for both buffers. The size - offset clamp also
    // prevents the last sizes entry from overshooting the requested copy range.
    while (offset < size) {
        src_ptrs.push_back(src_ptr);
        dst_ptrs.push_back(dst_ptr);
        
        size_t advance = std::min({src_rem, dst_rem, size - offset});
        sizes.push_back(advance);

        offset += advance;
        src_rem -= advance;
        dst_rem -= advance;

        if (src_rem == 0 && offset < size) {
            auto s = block_bounds(*this, static_cast<size_t>(src_offset) + offset);
            src_ptr = s.data();
            src_rem = s.size();
        } else {
            src_ptr += advance;
        }

        if (dst_rem == 0 && offset < size) {
            auto s = block_bounds(dst, static_cast<size_t>(dst_offset) + offset);
            dst_ptr = s.data();
            dst_rem = s.size();
        } else {
            dst_ptr += advance;
        }
    }

    cuda_memcpy_batch_async(
        std::span<void const*>(src_ptrs),
        std::span<void const*>(dst_ptrs),
        std::span<std::size_t>(sizes),
        stream_
    );

    dst.latest_write_event().stream_wait(stream_);
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

    // // We have to sync both before *and* after the memcpy. Otherwise, `src.stream()`
    // // might deallocate `src` before the memcpy enqueued on `dst.stream()` has completed.
    // src.latest_write_event().stream_wait(dst.stream());
    // dst.write_access([&](std::byte* dst_data, rmm::cuda_stream_view stream) {
    //     RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
    //         dst_data + dst_offset,
    //         src.data() + src_offset,
    //         size,
    //         cudaMemcpyDefault,
    //         stream
    //     ));
    // });
    // // after the dst.write_access(), its last_write_event is recorded on dst.stream(). So,
    // // we need the src.stream() to wait for that event.
    // dst.latest_write_event().stream_wait(src.stream());
    src.copy_to(dst, size, dst_offset, src_offset);
}

}  // namespace rapidsmpf
