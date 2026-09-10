/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include <kvikio/compat_mode.hpp>
#include <kvikio/file_handle.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::disk {

namespace {

std::size_t wait_io(kvikio::FileHandle& file, std::future<std::size_t> io) {
    auto const n = io.get();
    file.close();
    return n;
}

}  // namespace

std::filesystem::path DiskResource::create_unique_path() const {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    RAPIDSMPF_EXPECTS(
        !ec,
        "failed to create directory " + dir_.string() + ": " + ec.message(),
        std::runtime_error
    );

    auto path_template = (dir_ / "XXXXXX").string();
    auto const fd = ::mkstemp(path_template.data());
    auto const open_error = errno;
    RAPIDSMPF_EXPECTS(
        fd >= 0,
        "failed to reserve a unique file under " + dir_.string() + ": "
            + std::string{std::strerror(open_error)},
        std::runtime_error
    );
    if (::close(fd) != 0) {
        auto const err = errno;
        std::error_code remove_error;
        std::filesystem::remove(path_template, remove_error);
        RAPIDSMPF_FAIL(
            "close after file reservation failed: " + std::string{std::strerror(err)},
            std::runtime_error
        );
    }
    return path_template;
}

std::size_t DiskResource::write(
    std::filesystem::path const& path,
    void const* data,
    std::size_t size,
    [[maybe_unused]] MemoryType mem_type,
    std::size_t file_offset
) const {
    kvikio::FileHandle file{
        path.string(), "w+", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    };
    return wait_io(
        file,
        file.pwrite(
            data,
            size,
            file_offset,
            kvikio::defaults::task_size(),
            kvikio::defaults::gds_threshold(),
            false  // sync_default_stream
        )
    );
}

std::size_t DiskResource::read(
    std::filesystem::path const& path,
    void* data,
    std::size_t size,
    [[maybe_unused]] MemoryType mem_type,
    std::size_t file_offset
) const {
    kvikio::FileHandle file{
        path.string(), "r", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    };
    return wait_io(
        file,
        file.pread(
            data,
            size,
            file_offset,
            kvikio::defaults::task_size(),
            kvikio::defaults::gds_threshold(),
            false  // sync_default_stream
        )
    );
}

void DiskResource::flush(std::filesystem::path const& path) const {
    auto const fd = ::open(path.c_str(), O_RDONLY);
    RAPIDSMPF_EXPECTS(
        fd >= 0,
        "open for fdatasync failed: " + std::string{std::strerror(errno)},
        std::runtime_error
    );
    if (::fdatasync(fd) != 0) {
        auto const error = std::string{std::strerror(errno)};
        ::close(fd);
        RAPIDSMPF_FAIL("fdatasync failed: " + error, std::runtime_error);
    }
    RAPIDSMPF_EXPECTS(
        ::close(fd) == 0,
        "close after fdatasync failed: " + std::string{std::strerror(errno)},
        std::runtime_error
    );
}

std::optional<std::filesystem::path> spill_dir_from_options(config::Options options) {
    return options.get<std::optional<std::filesystem::path>>(
        "disk_spill_dir",
        [](std::string const& value) -> std::optional<std::filesystem::path> {
            auto parsed = parse_optional(value);
            if (!parsed.has_value()) {
                return std::nullopt;
            }
            RAPIDSMPF_EXPECTS(
                !trim(*parsed).empty(),
                "`disk_spill_dir` must be a non-empty path",
                std::invalid_argument
            );
            return std::filesystem::path{*parsed};
        }
    );
}

}  // namespace rapidsmpf::disk
