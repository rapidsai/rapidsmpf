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

void HostBuffer::deallocate_async() noexcept {
    if (!span_.empty()) {
        mr_.deallocate(stream_, span_.data(), span_.size());
    }
    span_ = {};
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : stream_{other.stream_}, mr_{other.mr_}, span_{std::exchange(other.span_, {})} {}

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
};

}  // namespace rapidsmpf
