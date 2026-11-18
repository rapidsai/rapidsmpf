/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <cuda/stream>

#include <rapidsmpf/disk/disk_resource.hpp>

namespace rapidsmpf {

class Buffer;
class Statistics;

/**
 * @brief File-backed handle to a byte buffer.
 *
 * `DiskBuffer` owns a backing file and deletes it on `deallocate()` or
 * destruction. Disk-backed `Buffer` objects store a `DiskBuffer` as their
 * storage.
 *
 * @note The backing file is owned by the `DiskBuffer` and its path is not exposed, so the
 * file can only change through this class (or `buffer_copy()`, which is a friend).
 * `release()` hands the file over to the caller.
 */
class DiskBuffer {
    friend void buffer_copy(
        std::shared_ptr<Statistics> statistics,
        Buffer& dst,
        Buffer const& src,
        std::size_t size,
        std::ptrdiff_t dst_offset,
        std::ptrdiff_t src_offset
    );

  public:
    /**
     * @brief Create a file-backed buffer.
     *
     * @param disk Disk resource used to create and access the backing file.
     * @param size Logical buffer size in bytes.
     * @param stream CUDA stream associated with the buffer.
     */
    DiskBuffer(
        std::shared_ptr<DiskResource> disk, std::size_t size, cuda::stream_ref stream
    );

    /**
     * @brief Create a buffer that takes ownership of an existing file.
     *
     * The file is placed under @p disk's directory with `DiskResource::adopt()`, which
     * copies it if it is on a different filesystem. The logical size is the file's size.
     * Like any backing file, it is deleted when the buffer is deallocated or destroyed.
     *
     * @param disk Disk resource used to access the backing file.
     * @param path Existing regular file to take ownership of.
     * @param stream CUDA stream associated with the buffer.
     * @throws std::invalid_argument if @p path is not an existing regular file.
     * @throws std::runtime_error if the file cannot be placed under @p disk's directory.
     * @throws std::filesystem::filesystem_error if the file's size cannot be queried. The
     *         file is deleted in that case.
     */
    DiskBuffer(
        std::shared_ptr<DiskResource> disk,
        std::filesystem::path path,
        cuda::stream_ref stream
    );

    ~DiskBuffer();

    /// @brief Move constructor.
    /// @param other Buffer to move from.
    DiskBuffer(DiskBuffer&& other) noexcept;

    /**
     * @brief Move assignment.
     *
     * Deletes this buffer's current backing file (if any), then takes over @p other's
     * disk resource and backing file in place. No file data is copied. @p other is left
     * without a backing file and with a zero size. To place the file under a different
     * disk resource first, call `other.relocate(...)` before assigning.
     *
     * @param other Buffer to move from.
     * @return Reference to this buffer.
     */
    DiskBuffer& operator=(DiskBuffer&& other) noexcept;
    DiskBuffer(DiskBuffer const&) = delete;  ///< Not copyable.
    DiskBuffer& operator=(DiskBuffer const&) = delete;  ///< Not copy-assignable.

    /**
     * @brief Logical size of the buffer.
     *
     * @return Buffer size in bytes.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Current size of the backing file.
     *
     * @return Backing file size in bytes.
     * @throws std::logic_error if the buffer has no backing file (after `release()` or
     * `deallocate()`).
     * @throws std::filesystem::filesystem_error if the size cannot be queried, e.g. the
     *         file was removed.
     * @throws std::overflow_error if the size does not fit in `std::size_t`.
     */
    [[nodiscard]] std::size_t file_size() const;

    /**
     * @brief Whether the buffer holds no data.
     *
     * A buffer is empty if its logical `size() == 0` or if its backing file is empty,
     * e.g. a freshly created buffer that has not been written to yet.
     *
     * @return `true` if the buffer holds no data.
     * @throws See `file_size()`, which is queried when `size()` is non-zero.
     */
    [[nodiscard]] bool is_empty() const;

    /**
     * @brief Copy the backing file contents into a host `std::vector`.
     *
     * This is primarily intended for debugging or testing.
     *
     * @return A vector containing the backing file bytes.
     * @throws See `file_size()`.
     */
    [[nodiscard]] std::vector<std::uint8_t> copy_to_uint8_vector() const;

    /**
     * @brief Disk resource used for backing-file I/O.
     *
     * @return Shared pointer to the disk resource.
     */
    [[nodiscard]] std::shared_ptr<DiskResource> const& disk_resource() const noexcept {
        return disk_;
    }

    /**
     * @brief Get the CUDA stream associated with this buffer.
     *
     * @return CUDA stream reference.
     */
    [[nodiscard]] cuda::stream_ref stream() const noexcept {
        return stream_;
    }

    /**
     * @brief Set the CUDA stream associated with this buffer.
     *
     * @param stream New CUDA stream.
     */
    void set_stream(cuda::stream_ref stream) noexcept;

    /**
     * @brief Delete the backing file, if any.
     *
     * After deallocation the buffer has no backing file. Safe to call multiple times.
     */
    void deallocate() noexcept;

    /**
     * @brief Release ownership of the backing file without deleting it.
     *
     * After this call the buffer has no backing file, `size()` is zero, and this buffer
     * no longer deletes the file. The caller owns it. Returns an empty path if there is
     * no backing file.
     *
     * @return Path of the released backing file.
     */
    [[nodiscard]] std::filesystem::path release() noexcept;

    /**
     * @brief Move the backing file under another disk resource.
     *
     * Uses `DiskResource::adopt()`, which copies the file when @p disk is on a different
     * filesystem. If relocation fails, this buffer is left unchanged.
     *
     * @param disk Disk resource to move the backing file to.
     * @throws std::invalid_argument if the backing file no longer exists.
     * @throws std::runtime_error if the file cannot be relocated.
     */
    void relocate(std::shared_ptr<DiskResource> disk);

  private:
    ///@brief Path to the backing file. Private so that nobody outside the buffer can
    /// modify or delete the file.
    [[nodiscard]] std::filesystem::path const& path() const noexcept {
        return path_;
    }

    std::shared_ptr<DiskResource> disk_;
    std::filesystem::path path_;
    std::size_t size_;
    cuda::stream_ref stream_;
};

}  // namespace rapidsmpf
