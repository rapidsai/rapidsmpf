/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <utility>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::disk {

DiskBuffer::DiskBuffer(
    std::shared_ptr<DiskResource> disk, std::filesystem::path path, std::size_t size
)
    : disk_{std::move(disk)}, path_{std::move(path)}, size_{size} {}

DiskBuffer::DiskBuffer(DiskBuffer&& other) noexcept
    : disk_{std::move(other.disk_)},
      path_{std::move(other.path_)},
      size_{std::exchange(other.size_, 0)} {}

void DiskBuffer::deallocate() noexcept {
    if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        path_.clear();
    }
    size_ = 0;
    disk_.reset();
}

DiskBuffer::~DiskBuffer() {
    deallocate();
}

std::unique_ptr<DiskBuffer> DiskBuffer::from_buffer(
    std::unique_ptr<Buffer> source, BufferResource& br
) {
    RAPIDSMPF_EXPECTS(
        source != nullptr, "source buffer cannot be null", std::logic_error
    );

    auto disk = br.disk_resource();
    auto path = disk->create_unique_path();
    auto const nbytes = source->size;

    auto make_disk_buffer = [&](std::size_t size) {
        return std::unique_ptr<DiskBuffer>(
            new DiskBuffer{std::move(disk), std::move(path), size}
        );
    };

    if (nbytes == 0) {
        return make_disk_buffer(0);
    }

    RAPIDSMPF_EXPECTS(
        source->is_latest_write_done(),
        "cannot write buffer to disk with pending stream-ordered writes",
        std::logic_error
    );

    try {
        auto const transferred =
            disk->write(path, source->data(), nbytes, source->mem_type())->get();
        RAPIDSMPF_EXPECTS(
            transferred == nbytes,
            "disk write transferred " + format_nbytes(transferred) + " of "
                + format_nbytes(nbytes),
            std::runtime_error
        );
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        throw;
    }

    return make_disk_buffer(nbytes);
}

std::unique_ptr<Buffer> DiskBuffer::restore(
    std::unique_ptr<DiskBuffer> source,
    MemoryReservation& reservation,
    cuda::stream_ref stream
) {
    RAPIDSMPF_EXPECTS(
        source != nullptr, "source disk buffer cannot be null", std::logic_error
    );
    RAPIDSMPF_EXPECTS(
        reservation.size() >= source->size(),
        "MemoryReservation(" + format_nbytes(reservation.size()) + ") isn't big enough ("
            + format_nbytes(source->size()) + ")",
        rapidsmpf::reservation_error
    );
    RAPIDSMPF_EXPECTS(reservation.br() != nullptr, "reservation has no BufferResource");
    RAPIDSMPF_EXPECTS(source->disk_ != nullptr, "DiskBuffer has no DiskResource");

    auto buffer = reservation.br()->make_buffer(source->size(), stream, reservation);

    if (source->size() == 0) {
        return buffer;
    }

    buffer->write_access([&](std::byte* ptr, cuda::stream_ref buf_stream) {
        RAPIDSMPF_EXPECTS(
            buf_stream.get() == stream.get(),
            "destination buffer stream does not match restore stream",
            std::logic_error
        );
        auto const transferred =
            source->disk_->read(source->path_, ptr, source->size(), buffer->mem_type())
                ->get();
        RAPIDSMPF_EXPECTS(
            transferred == source->size(),
            "disk read transferred " + format_nbytes(transferred) + " of "
                + format_nbytes(source->size()),
            std::runtime_error
        );
    });
    stream.sync();

    return buffer;
}

}  // namespace rapidsmpf::disk
