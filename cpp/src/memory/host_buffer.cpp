/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <cuda/stream>

#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/resource_types.hpp>

namespace rapidsmpf {

HostBuffer::HostBuffer(
    std::size_t size,
    cuda::stream_ref stream,
    cuda::mr::any_resource<cuda::mr::host_accessible> mr
)
    : stream_{stream} {
    if (size > 0) {
        auto* ptr = static_cast<std::byte*>(
            mr.allocate(stream_, size, alignof(::cuda::std::max_align_t))
        );
        span_ = std::span<std::byte>{ptr, size};
        deallocate_fn_ = [mr = std::move(mr), ptr, size](cuda::stream_ref s) mutable {
            mr.deallocate(s, ptr, size, alignof(::cuda::std::max_align_t));
        };
    }
}

HostBuffer::HostBuffer(
    std::span<std::byte> span,
    cuda::stream_ref stream,
    std::function<void(cuda::stream_ref)> deallocate_fn
)
    : stream_{stream}, span_{span}, deallocate_fn_{std::move(deallocate_fn)} {}

void HostBuffer::deallocate_async() noexcept {
    if (!span_.empty() && deallocate_fn_) {
        deallocate_fn_(stream_);
        deallocate_fn_ = nullptr;
    }
    span_ = {};
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : stream_{other.stream_},
      span_{std::exchange(other.span_, {})},
      deallocate_fn_{std::exchange(other.deallocate_fn_, {})} {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) {
    if (this != &other) {
        RAPIDSMPF_EXPECTS(
            span_.empty(),
            "cannot move into an already initialized HostBuffer",
            std::invalid_argument
        );
        stream_ = other.stream_;
        span_ = std::exchange(other.span_, {});
        deallocate_fn_ = std::exchange(other.deallocate_fn_, {});
    }
    return *this;
}

HostBuffer::~HostBuffer() noexcept {
    deallocate_async();
}

cuda::stream_ref HostBuffer::stream() const noexcept {
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

void HostBuffer::set_stream(cuda::stream_ref new_stream) {
    stream_ = new_stream;
}

std::vector<std::uint8_t> HostBuffer::copy_to_uint8_vector() const {
    std::vector<std::uint8_t> ret(size());
    if (!empty()) {
        stream_.sync();
        RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(ret.data(), data(), size(), stream_));
        stream_.sync();
    }
    return ret;
};

HostBuffer HostBuffer::from_uint8_vector(
    std::vector<std::uint8_t> const& data,
    cuda::stream_ref stream,
    rmm::host_async_resource_ref mr
) {
    HostBuffer ret(data.size(), stream, mr);
    if (!ret.empty()) {
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(ret.data(), data.data(), data.size(), stream)
        );
        stream.sync();  // need to ensure that data outlives the async copy
    }
    return ret;
}

HostBuffer HostBuffer::from_owned_vector(
    std::vector<std::uint8_t>&& data, cuda::stream_ref stream
) {
    // Wrap in shared_ptr so the lambda is copyable (required by std::function).
    auto shared_vec = std::make_shared<std::vector<std::uint8_t>>(std::move(data));
    auto* ptr = reinterpret_cast<std::byte*>(shared_vec->data());
    std::span<std::byte> span{ptr, shared_vec->size()};

    return HostBuffer{
        span, stream, [shared_vec_ = std::move(shared_vec)](cuda::stream_ref) mutable {
            shared_vec_.reset();
        }
    };
}

HostBuffer HostBuffer::from_rmm_device_buffer(
    std::unique_ptr<rmm::device_buffer> pinned_host_buffer, cuda::stream_ref stream
) {
    RAPIDSMPF_EXPECTS(
        pinned_host_buffer != nullptr,
        "pinned_host_buffer must not be null",
        std::invalid_argument
    );

    RAPIDSMPF_EXPECTS(
        is_host_accessible(pinned_host_buffer->memory_resource()),
        "pinned_host_buffer's memory resource must be host accessible",
        std::logic_error
    );

    // Wrap in shared_ptr so the lambda is copyable (required by std::function).
    auto shared_db = std::make_shared<rmm::device_buffer>(std::move(*pinned_host_buffer));
    std::span<std::byte> span{
        static_cast<std::byte*>(shared_db->data()), shared_db->size()
    };

    return HostBuffer{
        std::move(span),
        stream,
        [shared_db_ = std::move(shared_db)](cuda::stream_ref) mutable {
            shared_db_.reset();
        }
    };
}

}  // namespace rapidsmpf
