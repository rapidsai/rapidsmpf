/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/detail/aligned_buffer.hpp>

namespace rapidsmpf::detail {

AlignedBuffer::AlignedBuffer(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::size_t size,
    std::size_t alignment
)
    : mr_{mr},
      stream_{stream},
      size_{size},
      alignment_{alignment},
      data_{mr_.allocate(stream_, size_, alignment_)} {}

/**
 * @brief Deallocate the buffer.
 */
AlignedBuffer::~AlignedBuffer() noexcept {
    mr_.deallocate(stream_, data_, size_, alignment_);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : mr_{other.mr_},
      stream_{other.stream_},
      size_{other.size_},
      alignment_{other.alignment_},
      data_{std::exchange(other.data_, nullptr)} {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) {
    if (this != &other) {
        RAPIDSMPF_EXPECTS(
            !data_,
            "cannot move into an already initialized aligned_buffer",
            std::invalid_argument
        );
    }
    mr_ = other.mr_;
    stream_ = other.stream_;
    size_ = other.size_;
    alignment_ = other.alignment_;
    data_ = std::exchange(other.data_, nullptr);
    return *this;
}

rmm::cuda_stream_view AlignedBuffer::stream() const noexcept {
    return stream_;
};

void* AlignedBuffer::data() noexcept {
    return data_;
};

std::size_t AlignedBuffer::size() const noexcept {
    return size_;
};

rmm::device_async_resource_ref AlignedBuffer::mr() noexcept {
    return mr_;
};

}  // namespace rapidsmpf::detail
