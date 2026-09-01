/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cstring>
#include <future>
#include <memory>
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

namespace rapidsmpf::disk {

namespace {

class KvikioDiskFuture final : public DiskFuture {
  public:
    KvikioDiskFuture(
        std::unique_ptr<kvikio::FileHandle> file, std::future<std::size_t> io
    )
        : file_{std::move(file)}, io_{std::move(io)} {}

    ~KvikioDiskFuture() override {
        if (io_.valid()) {
            try {
                std::ignore = get();
            } catch (...) {
            }
        }
    }

    [[nodiscard]] bool valid() const noexcept override {
        return io_.valid();
    }

    [[nodiscard]] std::size_t get() override {
        auto const n = io_.get();
        file_->close();
        return n;
    }

  private:
    std::unique_ptr<kvikio::FileHandle> file_;
    std::future<std::size_t> io_;
};

}  // namespace

std::unique_ptr<DiskFuture> DiskResource::write(
    std::filesystem::path const& path,
    void const* data,
    std::size_t size,
    [[maybe_unused]] MemoryType mem_type,
    std::size_t file_offset
) {
    auto file = std::make_unique<kvikio::FileHandle>(
        path.string(), "w+", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    );
    auto io = file->pwrite(
        data,
        size,
        file_offset,
        kvikio::defaults::task_size(),
        kvikio::defaults::gds_threshold(),
        false  // sync_default_stream
    );
    return std::make_unique<KvikioDiskFuture>(std::move(file), std::move(io));
}

std::unique_ptr<DiskFuture> DiskResource::read(
    std::filesystem::path const& path,
    void* data,
    std::size_t size,
    [[maybe_unused]] MemoryType mem_type,
    std::size_t file_offset
) {
    auto file = std::make_unique<kvikio::FileHandle>(
        path.string(), "r", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    );
    auto io = file->pread(
        data,
        size,
        file_offset,
        kvikio::defaults::task_size(),
        kvikio::defaults::gds_threshold(),
        false  // sync_default_stream
    );
    return std::make_unique<KvikioDiskFuture>(std::move(file), std::move(io));
}

void DiskResource::flush(std::filesystem::path const& path) {
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

std::filesystem::path default_spill_directory(config::Options options) {
    return options.get<std::filesystem::path>(
        "disk_spill_dir", [](std::string const& value) {
            if (value.empty()) {
                return std::filesystem::temp_directory_path();
            }
            return std::filesystem::path{value};
        }
    );
}

}  // namespace rapidsmpf::disk
