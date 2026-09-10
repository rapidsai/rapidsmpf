/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {

class BufferResource;

namespace disk {

/**
 * @brief Non-stream-ordered disk I/O for host or device byte buffers.
 *
 * Uses KvikIO with CompatMode::AUTO (GDS when available, POSIX/compat otherwise).
 *
 * Callers must synchronize any CUDA stream that produced or consumes a device
 * pointer before calling write() or read(). KvikIO is not asked to synchronize
 * the default stream (`sync_default_stream=false`).
 *
 * Disk I/O is intentionally outside the MemoryType / BufferResource taxonomy.
 * `BufferResource` owns a `std::shared_ptr<DiskResource>` when a spill
 * directory is configured; `DiskBuffer`s hold additional copies so the
 * resource outlives those buffers.
 */
class DiskResource : public BackRefMixin<BufferResource> {
  public:
    ~DiskResource() = default;

    DiskResource(DiskResource const&) = delete;
    DiskResource& operator=(DiskResource const&) = delete;
    DiskResource(DiskResource&&) = delete;
    DiskResource& operator=(DiskResource&&) = delete;

    /**
     * @brief Directory used for file creation.
     *
     * @return Configured directory path.
     */
    [[nodiscard]] std::filesystem::path const& directory() const noexcept {
        return dir_;
    }

    /**
     * @brief Reserve a unique file path under `directory()`.
     *
     * Atomically creates an empty file using `mkstemp`.
     *
     * @return Path to the reserved empty file.
     */
    [[nodiscard]] std::filesystem::path create_unique_path() const;

    /**
     * @brief Write bytes to a file and block until the transfer completes.
     *
     * @param path File path.
     * @param data Host or device pointer to the source bytes. Must remain
     *        valid until this call returns.
     * @param size Number of bytes to write.
     * @param mem_type Memory type of @p data.
     * @param file_offset Byte offset within the file.
     * @return Number of bytes transferred. The caller must check this against
     *         @p size.
     */
    [[nodiscard]] std::size_t write(
        std::filesystem::path const& path,
        void const* data,
        std::size_t size,
        MemoryType mem_type,
        std::size_t file_offset = 0
    ) const;

    /**
     * @brief Read bytes from a file and block until the transfer completes.
     *
     * @param path File path.
     * @param data Host or device pointer to the destination buffer. Must remain
     *        valid until this call returns.
     * @param size Number of bytes to read.
     * @param mem_type Memory type of @p data.
     * @param file_offset Byte offset within the file.
     * @return Number of bytes transferred. The caller must check this against
     *         @p size.
     */
    [[nodiscard]] std::size_t read(
        std::filesystem::path const& path,
        void* data,
        std::size_t size,
        MemoryType mem_type,
        std::size_t file_offset = 0
    ) const;

    /**
     * @brief Durably synchronize file data to storage (fdatasync).
     *
     * Not used on the default spill path; exposed for benchmark durability cases.
     *
     * @param path File path.
     */
    void flush(std::filesystem::path const& path) const;

    /**
     * @brief Compare two disk resources.
     *
     * @param other Resource to compare with.
     * @return `true` if both have the same directory and back-reference state.
     */
    [[nodiscard]] bool operator==(DiskResource const& other) const noexcept = default;

  private:
    explicit DiskResource(std::filesystem::path dir) : dir_{std::move(dir)} {}

    friend class rapidsmpf::BufferResource;

    std::filesystem::path dir_;
};

/**
 * @brief Spill directory from `disk_spill_dir` (`RAPIDSMPF_DISK_SPILL_DIR`).
 *
 * Disabled values (`false`, `none`, …) yield `std::nullopt`. An empty
 * string is rejected.
 *
 * @param options Configuration options.
 * @return Configured directory, if disk spilling is enabled.
 * @throws std::invalid_argument if the option is an empty string.
 */
[[nodiscard]] std::optional<std::filesystem::path> spill_dir_from_options(
    config::Options options
);

}  // namespace disk
}  // namespace rapidsmpf
