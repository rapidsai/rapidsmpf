/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/memory/host_buffer.hpp>

namespace rapidsmpf {

HostBuffer::HostBuffer(
    std::size_t size, rmm::cuda_stream_view stream, rmm::host_async_resource_ref mr
)
    : stream_{stream}, mr_{std::move(mr)} {
    if (size > 0) {
        auto* ptr = static_cast<std::byte*>(mr_.allocate(stream_, size));
        span_ = std::span<std::byte>{ptr, size};
    }
}

HostBuffer::HostBuffer(
    std::span<std::byte> span,
    rmm::cuda_stream_view stream,
    rmm::host_async_resource_ref mr,
    std::unique_ptr<void, OwnedStorageDeleter> owned_storage
)
    : stream_{stream}, mr_{mr}, span_{span}, owned_storage_{std::move(owned_storage)} {}

void HostBuffer::deallocate_async() noexcept {
    if (!span_.empty()) {
        // If we have owned storage, release it; otherwise deallocate via mr_.
        if (owned_storage_) {
            owned_storage_.reset();
        } else {
            mr_.deallocate(stream_, span_.data(), span_.size());
        }
    }
    span_ = {};
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : stream_{other.stream_},
      mr_{other.mr_},
      span_{std::exchange(other.span_, {})},
      owned_storage_{std::move(other.owned_storage_)} {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) {
    if (this != &other) {
        RAPIDSMPF_EXPECTS(
            span_.empty(),
            "cannot move into an already initialized HostBuffer",
            std::invalid_argument
        );
        stream_ = other.stream_;
        mr_ = other.mr_;
        span_ = std::exchange(other.span_, {});
        owned_storage_ = std::move(other.owned_storage_);
    }
    return *this;
}

HostBuffer::~HostBuffer() noexcept {
    deallocate_async();
}

rmm::cuda_stream_view HostBuffer::stream() const noexcept {
    return stream_;
}

std::size_t HostBuffer::size() const noexcept {
    return span_.size();
}

bool HostBuffer::empty() const noexcept {
    return span_.empty();
}

std::byte* HostBuffer::data() noexcept {
    return span_.data();
}

std::byte const* HostBuffer::data() const noexcept {
    return span_.data();
}

std::vector<std::uint8_t> HostBuffer::copy_to_uint8_vector() const {
    std::vector<std::uint8_t> ret(size());
    if (!empty()) {
        stream_.synchronize();
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(ret.data(), data(), size(), cudaMemcpyDefault, stream_)
        );
        stream_.synchronize();
    }
    return ret;
};

HostBuffer HostBuffer::from_uint8_vector(
    std::vector<std::uint8_t> const& data,
    rmm::cuda_stream_view stream,
    rmm::host_async_resource_ref mr
) {
    HostBuffer ret(data.size(), stream, mr);
    if (!ret.empty()) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            ret.data(), data.data(), data.size(), cudaMemcpyDefault, stream
        ));
    }
    return ret;
}

namespace {

/**
 * @brief A dummy host memory resource that terminates if used.
 *
 * This is used as a placeholder for `HostBuffer` instances that take ownership of
 * external storage. The resource is never actually used for allocation/deallocation
 * since such buffers use the `owned_storage_` deleter instead.
 */
class DummyHostMemoryResource final : public HostMemoryResource {
  public:
    void* allocate(rmm::cuda_stream_view, std::size_t, std::size_t) override {
        RAPIDSMPF_FAIL(
            "DummyHostMemoryResource should never be used for allocation",
            std::logic_error
        );
    }

    void deallocate(rmm::cuda_stream_view, void*, std::size_t, std::size_t) noexcept
        override {
        // This should never be called since buffers with owned_storage_ don't use mr_.
        // If we get here, something is seriously wrong - terminate.
        std::terminate();
    }
};

/// @brief Get a reference to the static dummy host memory resource.
rmm::host_async_resource_ref get_dummy_host_mr() {
    static DummyHostMemoryResource dummy_mr{};
    return dummy_mr;
}

}  // namespace

HostBuffer HostBuffer::from_owned_vector(
    std::vector<std::uint8_t>&& data, rmm::cuda_stream_view stream
) {
    // Get the data pointer and size before moving.
    // Moving a vector transfers ownership of the internal buffer but doesn't
    // invalidate the data pointer.
    auto* ptr = reinterpret_cast<std::byte*>(data.data());
    auto size = data.size();
    std::span<std::byte> span{ptr, size};

    // Move the vector into the lambda. The vector is destroyed when the deleter
    // (and its captured lambda) is destroyed.
    std::unique_ptr<void, OwnedStorageDeleter> owned_storage{
        ptr, [v = std::move(data)](void*) mutable { v.clear(); }
    };

    return HostBuffer{span, stream, get_dummy_host_mr(), std::move(owned_storage)};
}

HostBuffer HostBuffer::from_owned_device_buffer(
    std::unique_ptr<rmm::device_buffer> device_buffer, rmm::cuda_stream_view stream
) {
    RAPIDSMPF_EXPECTS(
        device_buffer != nullptr,
        "device_buffer must not be null",
        std::invalid_argument
    );

    auto* ptr = static_cast<std::byte*>(device_buffer->data());
    auto size = device_buffer->size();
    std::span<std::byte> span{ptr, size};

    // Move the device_buffer into the lambda. The buffer is destroyed when the
    // deleter (and its captured lambda) is destroyed.
    std::unique_ptr<void, OwnedStorageDeleter> owned_storage{
        ptr, [db = std::move(device_buffer)](void*) mutable { db.reset(); }
    };

    return HostBuffer{span, stream, get_dummy_host_mr(), std::move(owned_storage)};
}

}  // namespace rapidsmpf
