/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include <cuda/stream>

#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>

namespace rapidsmpf {

class BufferResource;

namespace disk {

/**
 * @brief File-backed handle to a byte buffer.
 *
 * `DiskBuffer` is intentionally separate from `Buffer` and outside the
 * `MemoryType` taxonomy. It owns a backing file and deletes it on
 * `deallocate()` or destruction.
 *
 * `from_buffer` and `restore` are synchronous: they block until the disk
 * transfer completes.
 */
class DiskBuffer {
  public:
    /**
     * @brief Move constructor.
     *
     * Transfers ownership of the backing file. The moved-from object is empty
     * (`path()` is empty and `size()` is zero).
     *
     * @param other Buffer to move from.
     */
    DiskBuffer(DiskBuffer&& other) noexcept;
    DiskBuffer& operator=(DiskBuffer&& other) = delete;  ///< Not move-assignable.

    ~DiskBuffer();

    DiskBuffer(DiskBuffer const&) = delete;  ///< Not copyable.
    DiskBuffer& operator=(DiskBuffer const&) = delete;  ///< Not copy-assignable.

    /**
     * @brief Payload size in bytes stored in the backing file.
     *
     * @return Size in bytes.
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Path to the backing file.
     *
     * @return Filesystem path to the backing file.
     */
    [[nodiscard]] std::filesystem::path const& path() const noexcept {
        return path_;
    }

    /**
     * @brief Delete the backing file, if any.
     *
     * After deallocation the buffer is empty (`path()` is empty and `size()`
     * is zero). Safe to call multiple times.
     */
    void deallocate() noexcept;

    /**
     * @brief Write the contents of @p source to a new file on disk.
     *
     * Takes ownership of @p source, creates a unique file under @p br's
     * configured disk directory, and blocks until the write completes. The
     * in-memory buffer is released after the write finishes. @p source must
     * already have no pending stream-ordered writes
     * (`is_latest_write_done()`).
     *
     * @param source Buffer whose bytes are written to disk.
     * @param br Buffer resource supplying disk I/O and directory configuration.
     * @return A disk-resident handle to the written bytes.
     */
    [[nodiscard]] static std::unique_ptr<DiskBuffer> from_buffer(
        std::unique_ptr<Buffer> source, BufferResource& br
    );

    /**
     * @brief Restore bytes from a disk handle into a newly allocated `Buffer`.
     *
     * Takes ownership of @p source, allocates from @p reservation on @p stream,
     * reads the backing file with blocking disk I/O, and records the fill as a
     * stream-ordered write on the destination buffer. The backing file is
     * deleted when @p source is destroyed.
     *
     * @param source Disk-resident handle whose bytes are restored.
     * @param reservation Memory reservation covering at least `source->size()`
     * bytes.
     * @param stream CUDA stream associated with the destination buffer.
     * @return In-memory buffer containing the file contents.
     */
    [[nodiscard]] static std::unique_ptr<Buffer> restore(
        std::unique_ptr<DiskBuffer> source,
        MemoryReservation& reservation,
        cuda::stream_ref stream
    );

  private:
    DiskBuffer(
        std::shared_ptr<DiskResource> disk, std::filesystem::path path, std::size_t size
    );

    std::shared_ptr<DiskResource> disk_;
    std::filesystem::path path_;
    std::size_t size_{};
};

}  // namespace disk
}  // namespace rapidsmpf
