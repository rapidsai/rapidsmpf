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
#include <tuple>
#include <utility>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
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

namespace {

/// Remove @p path if it exists, ignoring any error. Used to clean up after a failure.
void remove_ignoring_errors(std::filesystem::path const& path) noexcept {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace

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
        remove_ignoring_errors(path_template);
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
    std::filesystem::path const& src_path,
    std::filesystem::path const& dst_path,
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
    ScopedFd src{src_path, O_RDONLY};
    ScopedFd dst{dst_path, O_WRONLY | O_CREAT};
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


/// Removes a file when it goes out of scope, unless released. Cleans up after a failure.
class FileGuard {
  public:
    /// @param path The file to remove unless released.
    explicit FileGuard(std::filesystem::path path) noexcept : path_{std::move(path)} {}

    ~FileGuard() noexcept {
        if (!path_.empty()) {
            remove_ignoring_errors(path_);
        }
    }

    FileGuard(FileGuard const&) = delete;
    FileGuard& operator=(FileGuard const&) = delete;

    /// @return The guarded path.
    [[nodiscard]] std::filesystem::path const& path() const noexcept {
        return path_;
    }

    /// Keep the file. @return Its path.
    [[nodiscard]] std::filesystem::path release() noexcept {
        return std::exchange(path_, {});
    }

  private:
    std::filesystem::path path_;
};

/**
 * @brief Replace @p dst with a copy of @p src.
 *
 * Works across filesystems, unlike `rename(2)`, and unlike `copy_file_range(2)` on newer
 * kernels. The data is copied to a temporary sibling of @p dst and renamed into place,
 * so @p dst never holds a partial file. On failure the temporary file is removed.
 *
 * @throws std::runtime_error if the copy or rename fails.
 */
void copy_file_into(std::filesystem::path const& src, std::filesystem::path const& dst) {
    FileGuard tmp{dst.string() + ".copying"};
    std::error_code ec;
    std::filesystem::copy_file(
        src, tmp.path(), std::filesystem::copy_options::overwrite_existing, ec
    );
    if (!ec) {
        std::filesystem::rename(tmp.path(), dst, ec);
    }
    RAPIDSMPF_EXPECTS(
        !ec,
        "failed to copy " + src.string() + " to " + dst.string() + ": " + ec.message(),
        std::runtime_error
    );
    std::ignore = tmp.release();  // Renamed into place.
}

/**
 * @brief Try to make @p dst a copy-on-write reflink of @p src.
 *
 * @return False if the filesystem does not support reflinks between the two files.
 * @throws std::runtime_error on any other failure.
 */
bool try_reflink(std::filesystem::path const& src, std::filesystem::path const& dst) {
#ifdef FICLONE
    ScopedFd src_fd{src, O_RDONLY};
    ScopedFd dst_fd{dst, O_WRONLY | O_TRUNC};
    if (::ioctl(dst_fd.get(), FICLONE, src_fd.get()) == 0) {
        return true;
    }
    auto const err = errno;
    RAPIDSMPF_EXPECTS(
        err == EOPNOTSUPP || err == EXDEV || err == EINVAL || err == ENOTTY,
        "FICLONE of " + src.string() + " failed: " + std::strerror(err),
        std::runtime_error
    );
#endif
    return false;
}

}  // namespace

std::filesystem::path DiskResource::clone(
    std::filesystem::path const& path, cuda::stream_ref stream
) const {
    RAPIDSMPF_EXPECTS(
        std::filesystem::is_regular_file(path),
        "cannot clone " + path.string() + ": not a regular file",
        std::invalid_argument
    );
    stream.sync();
    FileGuard new_path{create_unique_path()};
    if (!try_reflink(path, new_path.path())) {
        copy_file_into(path, new_path.path());
    }
    return new_path.release();
}

std::filesystem::path DiskResource::adopt(std::filesystem::path const& path) const {
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
        return path;
    }
    // The reserved file is replaced by the rename, or by a copy across filesystems. The
    // source is removed only once the copy is in place, so a failure leaves it intact.
    FileGuard new_path{create_unique_path()};
    std::filesystem::rename(path, new_path.path(), ec);
    if (ec == std::errc::cross_device_link) {
        copy_file_into(path, new_path.path());
        std::filesystem::remove(path, ec);
    }
    RAPIDSMPF_EXPECTS(
        !ec,
        "failed to adopt " + path.string() + " into " + dir_.string() + ": "
            + ec.message(),
        std::runtime_error
    );
    return new_path.release();
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
