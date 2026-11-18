/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

DiskBuffer::DiskBuffer(
    std::shared_ptr<DiskResource> disk, std::size_t size, cuda::stream_ref stream
)
    : disk_{std::move(disk)}, size_{size}, stream_{stream} {
    RAPIDSMPF_EXPECTS(disk_ != nullptr, "disk resource cannot be null");
    path_ = disk_->create_unique_path();
}

DiskBuffer::DiskBuffer(
    std::shared_ptr<DiskResource> disk, FileHandle path, cuda::stream_ref stream
)
    : disk_{std::move(disk)}, size_{0}, stream_{stream} {
    RAPIDSMPF_EXPECTS(disk_ != nullptr, "disk resource cannot be null");
    path_ = disk_->adopt(path);
    // `path_` is owned from here on; if the size query throws, the destructor is not
    // run, so release the file explicitly.
    try {
        size_ = file_size();
    } catch (...) {
        deallocate();
        throw;
    }
}

DiskBuffer::DiskBuffer(DiskBuffer&& other) noexcept
    : disk_{std::move(other.disk_)},
      path_{std::exchange(other.path_, {})},
      size_{std::exchange(other.size_, 0)},
      stream_{other.stream_} {}

DiskBuffer& DiskBuffer::operator=(DiskBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        disk_ = std::move(other.disk_);
        size_ = other.size_;  // Before `release()`, which resets it.
        path_ = other.release();
        stream_ = other.stream_;
    }
    return *this;
}

FileHandle DiskBuffer::release() noexcept {
    size_ = 0;
    return std::exchange(path_, {});
}

void DiskBuffer::relocate(std::shared_ptr<DiskResource> disk) {
    RAPIDSMPF_EXPECTS(disk != nullptr, "disk resource cannot be null");
    if (!path_.empty()) {
        path_ = disk->adopt(path_);
    }
    disk_ = std::move(disk);
}

std::size_t DiskBuffer::file_size() const {
    return path_.file_size();
}

bool DiskBuffer::is_empty() const {
    return size_ == 0 || file_size() == 0;
}

std::vector<std::uint8_t> DiskBuffer::copy_to_uint8_vector() const {
    auto const size = file_size();
    RAPIDSMPF_EXPECTS(
        size <= size_,
        "DiskBuffer backing file is larger than the buffer size",
        std::runtime_error
    );
    std::vector<std::uint8_t> ret(size);
    if (size > 0) {
        auto const transferred = disk_->read(path_, ret.data(), size, stream_);
        RAPIDSMPF_EXPECTS(
            transferred == size,
            "failed to read the complete DiskBuffer backing file",
            std::runtime_error
        );
    }
    return ret;
}

void DiskBuffer::set_stream(cuda::stream_ref stream) noexcept {
    stream_ = stream;
}

void DiskBuffer::deallocate() noexcept {
    // `disk_` is null in a moved-from buffer, which has no file either.
    if (disk_ != nullptr) {
        disk_->remove(path_);
    }
}

DiskBuffer::~DiskBuffer() {
    deallocate();
}

}  // namespace rapidsmpf
