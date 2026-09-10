/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

namespace rapidsmpf {

class BufferResource;

namespace disk {

/**
 * @brief Future for a DiskResource read or write.
 *
 * I/O is submitted before the corresponding DiskResource method returns. The
 * pointer passed to write()/read() must remain valid until get() returns.
 *
 * Concrete implementations live in the translation unit and are returned as
 * `std::unique_ptr<DiskFuture>`.
 */
class DiskFuture {
  public:
    DiskFuture() = default;
    virtual ~DiskFuture() = default;
    DiskFuture(DiskFuture&&) = default;  ///< Movable.
    /**
     * @brief Move assignment.
     * @returns Moved this.
     */
    DiskFuture& operator=(DiskFuture&&) = default;
    DiskFuture(DiskFuture const&) = delete;  ///< Not copyable.
    DiskFuture& operator=(DiskFuture const&) = delete;  ///< Not copy-assignable.

    /**
     * @brief Whether this future refers to outstanding I/O.
     *
     * @return True until get() consumes the result.
     */
    [[nodiscard]] virtual bool valid() const noexcept = 0;

    /**
     * @brief Whether the transfer has completed without consuming the result.
     *
     * @return True if the I/O finished and `get()` has not been called yet.
     *
     * @note This is a point-in-time check and is subject to TOCTOU races:
     *       another thread may call `get()` after this returns `true`. Like
     *       `std::future`, concurrent `is_ready()`/`get()` from multiple
     *       threads is undefined behavior.
     */
    [[nodiscard]] virtual bool is_ready() const = 0;

    /**
     * @brief Wait for the transfer, release backend resources, and return the
     * byte count.
     *
     * @return Number of bytes transferred. The caller must check this against
     *         the requested size.
     *
     * @note Like `std::future::get()`, this is not thread-safe. Calling get()
     *       concurrently from multiple threads is undefined behavior.
     */
    [[nodiscard]] virtual std::size_t get() = 0;
};

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
 * `BufferResource` owns a `std::shared_ptr<DiskResource>`; `DiskBuffer`s hold
 * additional copies so the resource outlives those buffers.
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
     * @brief Write bytes to a file.
     *
     * @param path File path.
     * @param data Host or device pointer to the source bytes.
     * @param size Number of bytes to write.
     * @param mem_type Memory type of @p data.
     * @param file_offset Byte offset within the file.
     * @return Future that owns backend resources until it is waited.
     */
    [[nodiscard]] std::unique_ptr<DiskFuture> write(
        std::filesystem::path const& path,
        void const* data,
        std::size_t size,
        MemoryType mem_type,
        std::size_t file_offset = 0
    ) const;

    /**
     * @brief Read bytes from a file.
     *
     * @param path File path.
     * @param data Host or device pointer to the destination buffer.
     * @param size Number of bytes to read.
     * @param mem_type Memory type of @p data.
     * @param file_offset Byte offset within the file.
     * @return Future that owns backend resources until it is waited.
     */
    [[nodiscard]] std::unique_ptr<DiskFuture> read(
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
 * An empty option uses the system temporary directory.
 *
 * @param options Configuration options.
 * @return Directory used for spill files.
 */
[[nodiscard]] std::filesystem::path default_spill_directory(config::Options options);

}  // namespace disk
}  // namespace rapidsmpf
