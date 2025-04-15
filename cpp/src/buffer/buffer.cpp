/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdexcept>

#include <cuda/std/cstddef>

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

Buffer::Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br)
    : br{br},
      size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)} {
    RAPIDSMPF_EXPECTS(
        std::get<HostStorageT>(storage_) != nullptr, "the host_buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br)
    : br{br},
      size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)} {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceStorageT>(storage_) != nullptr, "the device buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

void* Buffer::data() {
    return std::visit([](auto& storage) -> void* { return storage->data(); }, storage_);
}

void const* Buffer::data() const {
    return std::visit(
        [](const auto& storage) -> void* { return storage->data(); }, storage_
    );
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
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        storage->data(), storage->size(), stream, br->device_mr()
                    ),
                    br
                });
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy(
    MemoryType target, rmm::cuda_stream_view stream
) const {
    if (mem_type() == target) {
        return copy(stream);
    }

    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) -> std::unique_ptr<Buffer> {
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        storage->data(), storage->size(), stream, br->device_mr()
                    ),
                    br
                });
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
                return std::unique_ptr<Buffer>(new Buffer{std::move(ret), br});
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    std::ptrdiff_t offset, std::ptrdiff_t length, rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(offset <= std::ptrdiff_t(size), "offset can't be more than size");
    RAPIDSMPF_EXPECTS(
        offset + length <= std::ptrdiff_t(size), "offset + length can't be more than size"
    );
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<std::vector<uint8_t>>(
                        storage->begin() + offset, storage->begin() + offset + length
                    ),
                    br
                });
            },
            [&](DeviceStorageT const& storage) {
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        static_cast<cuda::std::byte*>(storage->data()) + offset,
                        length,
                        stream,
                        br->device_mr()
                    ),
                    br
                });
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy_slice(
    MemoryType target,
    std::ptrdiff_t offset,
    std::ptrdiff_t length,
    rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(offset <= std::ptrdiff_t(size), "offset can't be more than size");
    RAPIDSMPF_EXPECTS(
        offset + length <= std::ptrdiff_t(size), "offset + length can't be more than size"
    );

    if (mem_type() == target) {
        return copy_slice(offset, length, stream);
    }

    // Implement the copy between each possible memory types (both directions).
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {  // host -> device
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<rmm::device_buffer>(
                        static_cast<uint8_t const*>(storage->data()) + offset,
                        length,
                        stream,
                        br->device_mr()
                    ),
                    br
                });
            },
            [&](DeviceStorageT const& storage) {  // device -> host
                {
                    auto ret = std::make_unique<std::vector<uint8_t>>(length);
                    RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                        ret->data(),
                        static_cast<cuda::std::byte const*>(storage->data()) + offset,
                        size_t(length),
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                    return std::unique_ptr<Buffer>(new Buffer{std::move(ret), br});
                }
            }
        },
        storage_
    );
}

std::ptrdiff_t Buffer::copy_to(
    Buffer& dest, std::ptrdiff_t offset, rmm::cuda_stream_view stream
) const {
    RAPIDSMPF_EXPECTS(mem_type() == dest.mem_type(), "buffer mem type mismatch");
    RAPIDSMPF_EXPECTS(
        dest.size >= (size_t(offset) + size), "offset can't be more than size"
    );
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) {
                std::memcpy(
                    static_cast<uint8_t*>(dest.data()) + offset, storage->data(), size
                );
                return std::ptrdiff_t(size);
            },
            [&](DeviceStorageT const& storage) {
                RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                    static_cast<cuda::std::byte*>(dest.data()) + offset,
                    storage->data(),
                    size,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
                return std::ptrdiff_t(size);
            }
        },
        storage_
    );
}

}  // namespace rapidsmpf
