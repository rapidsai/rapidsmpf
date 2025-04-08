/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {

namespace {
// Check that `ptr` isn't null.
template <typename T>
[[nodiscard]] std::unique_ptr<T> check_null(std::unique_ptr<T> ptr) {
    RAPIDSMP_EXPECTS(ptr, "unique pointer cannot be null", std::invalid_argument);
    return ptr;
}
}  // namespace

Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br)
    : host_buffer_{std::move(host_buffer)},
      mem_type{MemoryType::HOST},
      br{br},
      size{host_buffer_ ? host_buffer_->size() : 0} {
    RAPIDSMP_EXPECTS(host_buffer_ != nullptr, "the host_buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br)
    : device_buffer_{check_null(std::move(device_buffer))},
      mem_type{MemoryType::DEVICE},
      br{br},
      size{device_buffer_->size()} {
    RAPIDSMP_EXPECTS(device_buffer_ != nullptr, "the device buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

bool Buffer::is_moved() const noexcept {
    switch (mem_type) {
    case MemoryType::HOST:
        return host_buffer_ == nullptr;
    case MemoryType::DEVICE:
        return device_buffer_ == nullptr;
    }
    // This cannot happen, `mem_type` is always a member of `MemoryType`.
    return true;
}

void* Buffer::data() {
    switch (mem_type) {
    case MemoryType::HOST:
        return host()->data();
    case MemoryType::DEVICE:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

void const* Buffer::data() const {
    switch (mem_type) {
    case MemoryType::HOST:
        return host()->data();
    case MemoryType::DEVICE:
        return device()->data();
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    rmm::cuda_stream_view stream, size_t offset, size_t length
) const {
    RAPIDSMP_EXPECTS(offset <= size, "offset can't be more than  size");
    RAPIDSMP_EXPECTS(offset + length <= size, "offset + length cant be more than size");
    switch (mem_type) {
    case MemoryType::HOST:
        return std::make_unique<Buffer>(Buffer{
            std::make_unique<std::vector<uint8_t>>(
                *host()->begin() + offset, *host()->begin() + offset + length
            ),
            br
        });
    case MemoryType::DEVICE:
        return std::make_unique<Buffer>(Buffer{
            std::make_unique<rmm::device_buffer>(
                static_cast<cuda::std::byte const*>(device()->data()) + offset,
                length,
                stream,
                br->device_mr()
            ),
            br
        });
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    MemoryType target, rmm::cuda_stream_view stream, size_t offset, size_t length
) const {
    RAPIDSMP_EXPECTS(offset <= size, "offset can't be more than  size");
    RAPIDSMP_EXPECTS(offset + length <= size, "offset + length cant be more than size");

    if (mem_type == target) {
        return copy_slice(stream, offset, length);
    }

    // Implement the copy between each possible memory types (both directions).
    switch (mem_type) {
    case MemoryType::HOST:
        // host -> device
        return std::make_unique<Buffer>(Buffer{
            std::make_unique<rmm::device_buffer>(
                static_cast<uint8_t const*>(host()->data()) + offset,
                length,
                stream,
                br->device_mr()
            ),
            br
        });
    case MemoryType::DEVICE:
        // device -> host
        {
            auto ret = std::make_unique<std::vector<uint8_t>>(length);
            RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                ret->data(),
                static_cast<cuda::std::byte const*>(device()->data()) + offset,
                length,
                cudaMemcpyDeviceToHost,
                stream
            ));
            return std::make_unique<Buffer>(Buffer{std::move(ret), br});
        }
    }

    RAPIDSMP_FAIL("MemoryType: unknown");
}

size_t Buffer::copy_to(
    Buffer& target, size_t offset, rmm::cuda_stream_view stream
) const {
    RAPIDSMP_EXPECTS(target.size >= (offset + size), "offset can't be more than  size");
    switch (mem_type) {
    case MemoryType::HOST:
        std::memcpy(target.host()->data(), host()->data(), size);
        return size;
    case MemoryType::DEVICE:
        RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
            target.data(), device()->data(), size, cudaMemcpyDeviceToDevice, stream
        ));

        return size;
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy(rmm::cuda_stream_view stream) const {
    switch (mem_type) {
    case MemoryType::HOST:
        return std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(*host()), br}
        );
    case MemoryType::DEVICE:
        return std::make_unique<Buffer>(Buffer{
            std::make_unique<rmm::device_buffer>(
                device()->data(), device()->size(), stream, br->device_mr()
            ),
            br
        });
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

std::unique_ptr<Buffer> Buffer::copy(
    MemoryType target, rmm::cuda_stream_view stream
) const {
    // Implement the copy between each possible memory types (both directions).
    switch (mem_type) {
    case MemoryType::HOST:
        switch (target) {
        case MemoryType::HOST:  // host -> host
            return copy(stream);
        case MemoryType::DEVICE:  // host -> device
            return std::make_unique<Buffer>(Buffer{
                std::make_unique<rmm::device_buffer>(
                    host()->data(), host()->size(), stream, br->device_mr()
                ),
                br
            });
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    case MemoryType::DEVICE:
        switch (target) {
        case MemoryType::HOST:  // device -> host
            {
                auto ret = std::make_unique<std::vector<uint8_t>>(device()->size());
                RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                    ret->data(),
                    device()->data(),
                    device()->size(),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                return std::make_unique<Buffer>(Buffer{std::move(ret), br});
            }
        case MemoryType::DEVICE:  // device -> device
            return copy(stream);
        }
        RAPIDSMP_FAIL("MemoryType: unknown");
    }
    RAPIDSMP_FAIL("MemoryType: unknown");
}

}  // namespace rapidsmp
