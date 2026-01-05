/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

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
    : stream_{stream},
      mr_{std::move(mr)},
      span_{span},
      owned_storage_{std::move(owned_storage)} {}

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

HostBuffer HostBuffer::from_owned_vector(
    std::vector<std::uint8_t>&& data,
    rmm::cuda_stream_view stream,
    rmm::host_async_resource_ref mr
) {
    // Wrap in shared_ptr so the lambda is copyable (required by std::function).
    auto shared_vec = std::make_shared<std::vector<std::uint8_t>>(std::move(data));
    auto* ptr = reinterpret_cast<std::byte*>(shared_vec->data());
    auto size = shared_vec->size();
    std::span<std::byte> span{ptr, size};

    std::unique_ptr<void, OwnedStorageDeleter> owned_storage{
        ptr, [shared_vec_ = std::move(shared_vec)](void*) mutable { shared_vec_.reset(); }
    };

    return HostBuffer{span, stream, std::move(mr), std::move(owned_storage)};
}

HostBuffer HostBuffer::from_owned_rmm_pinned_host_buffer(
    std::unique_ptr<rmm::device_buffer> pinned_host_buffer,
    rmm::cuda_stream_view stream,
    PinnedMemoryResource& mr
) {
    RAPIDSMPF_EXPECTS(
        pinned_host_buffer != nullptr,
        "pinned_host_buffer must not be null",
        std::invalid_argument
    );

    RAPIDSMPF_EXPECTS(
        ptr_to_memory_type(pinned_host_buffer->data()) == MemoryType::PINNED_HOST,
        "pinned_host_buffer must be a pinned host buffer",
        std::invalid_argument
    );

    // Wrap in shared_ptr so the lambda is copyable (required by std::function).
    auto shared_db = std::make_shared<rmm::device_buffer>(std::move(*pinned_host_buffer));
    auto* ptr = static_cast<std::byte*>(shared_db->data());
    std::span<std::byte> span{ptr, shared_db->size()};

    std::unique_ptr<void, OwnedStorageDeleter> owned_storage{
        ptr, [shared_db_ = std::move(shared_db)](void*) mutable { shared_db_.reset(); }
    };

    return HostBuffer{std::move(span), stream, mr, std::move(owned_storage)};
}

}  // namespace rapidsmpf
