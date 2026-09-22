/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

DiskBuffer::DiskBuffer(
    std::shared_ptr<DiskResource> disk, std::size_t size, cuda::stream_ref stream
)
    : disk_{std::move(disk)}, size_{size}, stream_{stream} {
    RAPIDSMPF_EXPECTS(disk_ != nullptr, "disk resource cannot be null");
    path_ = disk_->create_unique_path();
}

DiskBuffer::DiskBuffer(DiskBuffer&& other) noexcept
    : disk_{std::move(other.disk_)},
      path_{std::exchange(other.path_, {})},
      size_{other.size_},
      stream_{other.stream_} {}

std::size_t DiskBuffer::file_size() const {
    return safe_cast<std::size_t>(std::filesystem::file_size(path_));
}

void DiskBuffer::read(
    Buffer& dst, std::size_t size, std::ptrdiff_t dst_offset, std::ptrdiff_t src_offset
) const {
    auto const backing_file_size = file_size();
    auto const offset = static_cast<std::size_t>(src_offset);
    RAPIDSMPF_EXPECTS(
        offset <= backing_file_size && size <= backing_file_size - offset,
        "src_offset + size can't be greater than the backing file size",
        std::invalid_argument
    );

    dst.write_access([&](std::byte* dst_data, cuda::stream_ref stream) {
        auto const transferred = disk_->read(
            path_, dst_data + dst_offset, size, dst.mem_type(), stream, src_offset
        );
        RAPIDSMPF_EXPECTS(
            transferred == size,
            "disk read transferred " + format_nbytes(transferred) + " of "
                + format_nbytes(size),
            std::runtime_error
        );
    });
}

void DiskBuffer::write(
    Buffer const& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset
) {
    auto const transferred = disk_->write(
        path_, src.data() + src_offset, size, src.mem_type(), stream_, dst_offset
    );
    RAPIDSMPF_EXPECTS(
        transferred == size,
        "disk write transferred " + format_nbytes(transferred) + " of "
            + format_nbytes(size),
        std::runtime_error
    );
}

std::vector<std::uint8_t> DiskBuffer::copy_to_uint8_vector() const {
    auto const size = file_size();
    std::vector<std::uint8_t> ret(size);
    if (size > 0) {
        auto const transferred =
            disk_->read(path_, ret.data(), size, MemoryType::HOST, stream_);
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
    if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        if (ec) {
            std::cerr << "Error removing DiskBuffer backing file '" << path_
                      << "': " << ec.message() << '\n';
        }
        path_.clear();
    }
}

DiskBuffer::~DiskBuffer() {
    deallocate();
}

}  // namespace rapidsmpf
