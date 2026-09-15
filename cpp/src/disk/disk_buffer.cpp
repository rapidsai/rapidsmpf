/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::disk {

DiskBuffer::DiskBuffer(std::shared_ptr<DiskResource> disk) : disk_{std::move(disk)} {
    RAPIDSMPF_EXPECTS(disk_ != nullptr, "disk resource cannot be null");
    path_ = disk_->create_unique_path();
}

DiskBuffer::DiskBuffer(DiskBuffer&& other) noexcept
    : disk_{std::move(other.disk_)}, path_{std::move(other.path_)} {}

std::uintmax_t DiskBuffer::file_size() const {
    return std::filesystem::file_size(path_);
}

std::vector<std::uint8_t> DiskBuffer::copy_to_uint8_vector() const {
    auto const size = safe_cast<std::size_t>(file_size());
    std::vector<std::uint8_t> ret(size);
    if (size > 0) {
        auto const transferred = disk_->read(path_, ret.data(), size, MemoryType::HOST);
        RAPIDSMPF_EXPECTS(
            transferred == size,
            "failed to read the complete DiskBuffer backing file",
            std::runtime_error
        );
    }
    return ret;
}

void DiskBuffer::deallocate() noexcept {
    if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        path_.clear();
    }
}

DiskBuffer::~DiskBuffer() {
    deallocate();
}

}  // namespace rapidsmpf::disk
