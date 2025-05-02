/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>
#include <stdexcept>
#include <utility>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

namespace {
// Check that `ptr` isn't null.
template <typename T>
[[nodiscard]] std::unique_ptr<T> check_null(std::unique_ptr<T> ptr) {
    RAPIDSMPF_EXPECTS(ptr, "unique pointer cannot be null", std::invalid_argument);
    return ptr;
}
}  // namespace

Buffer::Buffer(
    std::unique_ptr<std::vector<uint8_t>> host_buffer,
    BufferResource* br,
    std::shared_ptr<Event> event
)
    : br{br},
      size{host_buffer ? host_buffer->size() : 0},
      storage_{std::move(host_buffer)},
      event_{size > 0 ? std::move(event) : nullptr} {
    RAPIDSMPF_EXPECTS(
        std::get<HostStorageT>(storage_) != nullptr, "the host_buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
}

Buffer::Buffer(
    std::unique_ptr<rmm::device_buffer> device_buffer,
    BufferResource* br,
    std::shared_ptr<Event> event
)
    : br{br},
      size{device_buffer ? device_buffer->size() : 0},
      storage_{std::move(device_buffer)},
      event_{size > 0 ? std::move(event) : nullptr} {
    RAPIDSMPF_EXPECTS(
        std::get<DeviceStorageT>(storage_) != nullptr, "the device buffer cannot be NULL"
    );
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
    RAPIDSMPF_EXPECTS(
        size == 0 || event_, "non-zero sized device buffer must come with event."
    );
}

void* Buffer::data() {
    return std::visit([](auto&& storage) -> void* { return storage->data(); }, storage_);
}

void const* Buffer::data() const {
    return std::visit([](auto&& storage) -> void* { return storage->data(); }, storage_);
}

std::unique_ptr<Buffer> Buffer::copy(rmm::cuda_stream_view stream) const {
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) -> std::unique_ptr<Buffer> {
                // Have to ensure that any async work is complete.
                RAPIDSMPF_EXPECTS(
                    is_ready(), "Can't copy from host buffer with outstanding work"
                );
                return std::unique_ptr<Buffer>(new Buffer{
                    std::make_unique<std::vector<uint8_t>>(*storage), br, nullptr
                });
            },
            [&](const DeviceStorageT& storage) -> std::unique_ptr<Buffer> {
                auto event = std::make_shared<Event>();
                auto buf = std::make_unique<rmm::device_buffer>(
                    storage->data(), storage->size(), stream, br->device_mr()
                );
                event->record(stream);
                return std::unique_ptr<Buffer>(
                    new Buffer{std::move(buf), br, std::move(event)}
                );
            }
        },
        storage_
    );
}

std::unique_ptr<Buffer> Buffer::copy(MemoryType target, rmm::cuda_stream_view stream)
    const {
    if (mem_type() == target) {
        return copy(stream);
    }

    auto event = std::make_shared<Event>();
    return std::visit(
        overloaded{
            [&](const HostStorageT& storage) -> std::unique_ptr<Buffer> {
                auto buf = std::make_unique<rmm::device_buffer>(
                    storage->data(), storage->size(), stream, br->device_mr()
                );
                event->record(stream);

                return std::unique_ptr<Buffer>(
                    new Buffer{std::move(buf), br, std::move(event)}
                );
            },
            [&](const DeviceStorageT& storage) -> std::unique_ptr<Buffer> {
                auto ret = std::make_unique<std::vector<uint8_t>>(storage->size());
                RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
                    ret->data(),
                    storage->data(),
                    storage->size(),
                    cudaMemcpyDeviceToHost,
                    stream.value()
                ));
                event->record(stream);
                return std::unique_ptr<Buffer>(
                    new Buffer{std::move(ret), br, std::move(event)}
                );
            }
        },
        storage_
    );
}

bool Buffer::is_ready() const {
    return !event_ || event_->query();
}

}  // namespace rapidsmpf
