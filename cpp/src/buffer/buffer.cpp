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
    : br{br},
      size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)} {
    RAPIDSMP_EXPECTS(data() != nullptr, "the host_buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, BufferResource* br)
    : br{br},
      size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)} {
    RAPIDSMP_EXPECTS(data() != nullptr, "the device buffer cannot be NULL");
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

bool Buffer::is_moved() const noexcept {
    return std::visit(
        [](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return true;
            } else {
                // check either Storage is null
                return storage == nullptr;
            }
        },
        storage_
    );
}

void* Buffer::data() {
    return std::visit(
        [](auto&& storage) -> void* {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return nullptr;
            } else {
                return storage->data();
            }
        },
        storage_
    );
}

void const* Buffer::data() const {
    return std::visit(
        [](auto&& storage) -> void* {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return nullptr;
            } else {
                return storage->data();
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy(rmm::cuda_stream_view stream) const {
    return std::visit(
        [&](auto&& storage) -> std::unique_ptr<Buffer> {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, HostStorageT>) {
                return std::make_unique<Buffer>(
                    Buffer{std::make_unique<std::vector<uint8_t>>(*storage), br}
                );
            } else if constexpr (std::is_same_v<T, DeviceStorageT>) {
                return std::make_unique<Buffer>(Buffer{
                    std::make_unique<rmm::device_buffer>(
                        storage->data(), storage->size(), stream, br->device_mr()
                    ),
                    br
                });
            } else {
                RAPIDSMP_FAIL("Buffer is not initialized");
                return {};
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy(MemoryType target, rmm::cuda_stream_view stream)
    const {
    // Implement the copy between each possible memory types (both directions).
    switch (mem_type()) {
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
