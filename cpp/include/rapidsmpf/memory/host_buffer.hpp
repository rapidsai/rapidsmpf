/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>

namespace rapidsmpf {

class HostBuffer {
  public:
    HostBuffer(
        std::size_t size, rmm::cuda_stream_view stream, rmm::host_async_resource_ref mr
    )
        : stream_{stream}, mr_{mr} {
        if (size > 0) {
            auto* ptr = static_cast<std::byte*>(mr_.allocate(stream_, size));
            span_ = std::span<std::byte>{ptr, size};
        }
    }

    ~HostBuffer() noexcept {
        deallocate_async();
    }

    void deallocate_async() noexcept {
        if (!span_.empty()) {
            mr_.deallocate(stream_, span_.data(), span_.size());
        }
        span_ = {};
    }

    // Non copyable
    HostBuffer(HostBuffer const&) = delete;
    HostBuffer& operator=(HostBuffer const&) = delete;

    // Movable
    HostBuffer(HostBuffer&& other) noexcept
        : stream_{other.stream_}, mr_{other.mr_}, span_{std::exchange(other.span_, {})} {}

    HostBuffer& operator=(HostBuffer&& other) {
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

    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept {
        return stream_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return span_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return span_.empty();
    }

    [[nodiscard]] std::byte* data() noexcept {
        return span_.data();
    }

    [[nodiscard]] std::byte const* data() const noexcept {
        return span_.data();
    }

    // Mainly for debugging and testing, not performant.
    [[nodiscard]] std::vector<std::uint8_t> copy_to_uint8_vector() const {
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

  private:
    rmm::cuda_stream_view stream_;
    rmm::host_async_resource_ref mr_;
    std::span<std::byte> span_{};
};

}  // namespace rapidsmpf
