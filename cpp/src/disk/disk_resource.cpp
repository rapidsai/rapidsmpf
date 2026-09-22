/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

DiskResource::DiskResource(std::filesystem::path dir_prefix) {
    auto const parent = dir_prefix.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        RAPIDSMPF_EXPECTS(
            !ec,
            "failed to create directory " + parent.string() + ": " + ec.message(),
            std::runtime_error
        );
    }

    auto path_template = dir_prefix.string() + "-XXXXXX";
    auto const* path = ::mkdtemp(path_template.data());
    auto const create_error = errno;
    RAPIDSMPF_EXPECTS(
        path != nullptr,
        "failed to create a unique directory from " + path_template + ": "
            + std::string{std::strerror(create_error)},
        std::runtime_error
    );
    dir_ = path;
}

DiskResource::~DiskResource() noexcept {
    std::error_code ec;
    std::filesystem::remove(dir_, ec);
    if (ec) {
        std::cerr << "Error removing DiskResource directory '" << dir_
                  << "': " << ec.message() << '\n';
    }
}

std::filesystem::path DiskResource::create_unique_path() const {
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
    cuda::stream_ref stream,
    std::ptrdiff_t file_offset
) const {
    kvikio::FileHandle file{
        path.string(),
        std::filesystem::exists(path) ? "r+" : "w+",
        kvikio::FileHandle::m644,
        kvikio::CompatMode::AUTO
    };
    stream.sync();
    return file
        .pwrite(
            data,
            size,
            safe_cast<std::size_t>(file_offset),
            kvikio::defaults::task_size(),
            kvikio::defaults::gds_threshold(),
            false  // sync_default_stream
        )
        .get();
}

std::size_t DiskResource::read(
    std::filesystem::path const& path,
    void* data,
    std::size_t size,
    [[maybe_unused]] MemoryType mem_type,
    cuda::stream_ref stream,
    std::ptrdiff_t file_offset
) const {
    kvikio::FileHandle file{
        path.string(), "r", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    };
    stream.sync();
    return file
        .pread(
            data,
            size,
            safe_cast<std::size_t>(file_offset),
            kvikio::defaults::task_size(),
            kvikio::defaults::gds_threshold(),
            false  // sync_default_stream
        )
        .get();
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

}  // namespace rapidsmpf
