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
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/types.h>
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

std::size_t FileHandle::file_size() const {
    RAPIDSMPF_EXPECTS(!empty(), "cannot query the size of an empty FileHandle");
    return safe_cast<std::size_t>(std::filesystem::file_size(path_));
}

FileHandle DiskResource::create_unique_path() const {
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
    return FileHandle{std::filesystem::path{path_template}};
}

std::size_t DiskResource::write(
    FileHandle const& path,
    void const* data,
    std::size_t size,
    cuda::stream_ref stream,
    std::ptrdiff_t file_offset
) const {
    kvikio::FileHandle file{
        path.path_.string(),
        std::filesystem::exists(path.path_) ? "r+" : "w+",
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
    FileHandle const& path,
    void* data,
    std::size_t size,
    cuda::stream_ref stream,
    std::ptrdiff_t file_offset
) const {
    kvikio::FileHandle file{
        path.path_.string(), "r", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
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

namespace {

/// RAII file descriptor.
class ScopedFd {
  public:
    ScopedFd(std::filesystem::path const& path, int flags)
        : fd_{::open(path.c_str(), flags, 0644)} {
        RAPIDSMPF_EXPECTS(
            fd_ >= 0,
            "failed to open " + path.string() + ": " + std::string{std::strerror(errno)},
            std::runtime_error
        );
    }

    ~ScopedFd() noexcept {
        ::close(fd_);
    }

    ScopedFd(ScopedFd const&) = delete;
    ScopedFd& operator=(ScopedFd const&) = delete;

    [[nodiscard]] int get() const noexcept {
        return fd_;
    }

  private:
    int fd_;
};

}  // namespace

std::size_t DiskResource::copy(
    FileHandle const& src_path,
    FileHandle const& dst_path,
    std::size_t size,
    cuda::stream_ref stream,
    std::ptrdiff_t src_offset,
    std::ptrdiff_t dst_offset
) const {
    RAPIDSMPF_EXPECTS(
        src_path != dst_path,
        "source and destination paths must differ",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        src_offset >= 0 && dst_offset >= 0,
        "file offsets must be non-negative",
        std::invalid_argument
    );
    ScopedFd src{src_path.path_, O_RDONLY};
    ScopedFd dst{dst_path.path_, O_WRONLY | O_CREAT};
    stream.sync();

    auto in_off = static_cast<off_t>(src_offset);
    auto out_off = static_cast<off_t>(dst_offset);
    std::size_t done = 0;

    while (done < size) {
        auto const n =
            ::copy_file_range(src.get(), &in_off, dst.get(), &out_off, size - done, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            RAPIDSMPF_FAIL(
                "copy_file_range failed: " + std::string{std::strerror(errno)},
                std::runtime_error
            );
        }
        if (n == 0) {
            break;  // EOF on the source.
        }
        done += static_cast<std::size_t>(n);
    }
    return done;
}

namespace {

/**
 * @brief Replace @p dst_path with a copy of @p src_path, then remove @p src_path.
 *
 * Fallback for moving a file across filesystems, where `rename(2)` is not possible. The
 * data is first copied to a temporary sibling of @p dst_path and renamed into place, so
 * @p dst_path never holds a partial file.
 *
 * On failure the temporary file is removed and @p src_path is left in place. If only the
 * final removal of @p src_path fails, @p dst_path already holds the complete copy.
 *
 * @throws std::runtime_error if the copy, rename or removal fails.
 */
void copy_then_replace(
    std::filesystem::path const& src_path, std::filesystem::path const& dst_path
) {
    // Copy to a temporary sibling of the destination and rename it into place, so the
    // destination never holds a partial file.
    auto const tmp_path = std::filesystem::path{dst_path.string() + ".moving"};
    std::error_code ec;
    std::filesystem::copy_file(
        src_path, tmp_path, std::filesystem::copy_options::overwrite_existing, ec
    );
    if (!ec) {
        std::filesystem::rename(tmp_path, dst_path, ec);
    }
    if (!ec) {
        std::filesystem::remove(src_path, ec);
    }
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(tmp_path, ignored);
        RAPIDSMPF_FAIL(
            "failed to copy " + src_path.string() + " to " + dst_path.string() + ": "
                + ec.message(),
            std::runtime_error
        );
    }
}

}  // namespace

FileHandle DiskResource::adopt(std::filesystem::path const& path) const {
    RAPIDSMPF_EXPECTS(
        std::filesystem::is_regular_file(path),
        "cannot adopt " + path.string() + ": not a regular file",
        std::invalid_argument
    );
    // Compare directories by device and inode, not by path text, so that relative paths,
    // `./` and symlinked parents of `dir_` are recognized. On error `equivalent()`
    // returns false and we fall through to relocation, which is always safe.
    std::error_code ec;
    if (std::filesystem::equivalent(
            std::filesystem::absolute(path, ec).parent_path(), dir_, ec
        ))
    {
        return FileHandle{path};
    }

    // The reserved file is replaced by the rename (or copy) below.
    auto new_path = create_unique_path();
    auto cleanup = [&] { remove(new_path); };
    std::filesystem::rename(path, new_path.path_, ec);
    if (ec == std::errc::cross_device_link) {
        try {
            copy_then_replace(path, new_path.path_);
        } catch (...) {
            cleanup();
            throw;
        }
    } else if (ec) {
        cleanup();
        RAPIDSMPF_FAIL(
            "failed to adopt " + path.string() + " into " + dir_.string() + ": "
                + ec.message(),
            std::runtime_error
        );
    }
    return new_path;
}

FileHandle DiskResource::adopt(FileHandle const& path) const {
    RAPIDSMPF_EXPECTS(
        !path.empty(), "cannot adopt an empty FileHandle", std::invalid_argument
    );
    return adopt(path.path_);
}

bool DiskResource::remove(FileHandle& path) const noexcept {
    if (path.path_.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::remove(path.path_, ec);
    if (ec) {
        std::cerr << "Error removing file '" << path.path_ << "': " << ec.message()
                  << '\n';
    }
    path.path_.clear();
    return !ec;
}

void DiskResource::flush(FileHandle const& path) const {
    auto const fd = ::open(path.path_.c_str(), O_RDONLY);
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
