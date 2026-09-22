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

/**
 * @brief File-backed handle to a byte buffer.
 *
 * `DiskBuffer` owns a backing file and deletes it on `deallocate()` or
 * destruction. Disk-backed `Buffer` objects store a `DiskBuffer` as their
 * storage.
 */
class DiskBuffer {
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

    ~DiskBuffer();

    /// @brief Move constructor.
    /// @param other Buffer to move from.
    DiskBuffer(DiskBuffer&& other) noexcept;
    DiskBuffer& operator=(DiskBuffer&& other) = delete;  ///< Not move-assignable.
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
     */
    [[nodiscard]] std::size_t file_size() const;

    /**
     * @brief Read bytes from this disk buffer into an addressable buffer.
     *
     * @param dst Destination buffer.
     * @param size Number of bytes to read.
     * @param dst_offset Byte offset within @p dst.
     * @param src_offset Byte offset within this disk buffer's backing file.
     */
    void read(
        Buffer& dst,
        std::size_t size,
        std::ptrdiff_t dst_offset = 0,
        std::ptrdiff_t src_offset = 0
    ) const;

    /**
     * @brief Write bytes from an addressable buffer into this disk buffer.
     *
     * The source buffer's work must be ordered before this disk buffer's stream.
     *
     * @param src Source buffer.
     * @param size Number of bytes to write.
     * @param dst_offset Byte offset within this disk buffer's backing file.
     * @param src_offset Byte offset within @p src.
     */
    void write(
        Buffer const& src,
        std::size_t size,
        std::ptrdiff_t dst_offset = 0,
        std::ptrdiff_t src_offset = 0
    );

    /**
     * @brief Copy the backing file contents into a host `std::vector`.
     *
     * This is primarily intended for debugging or testing.
     *
     * @return A vector containing the backing file bytes.
     */
    [[nodiscard]] std::vector<std::uint8_t> copy_to_uint8_vector() const;

    /**
     * @brief Path to the backing file.
     *
     * @return Filesystem path to the backing file.
     */
    [[nodiscard]] std::filesystem::path const& path() const noexcept {
        return path_;
    }

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
     * After deallocation `path()` is empty. Safe to call multiple times.
     */
    void deallocate() noexcept;

  private:
    std::shared_ptr<DiskResource> disk_;
    std::filesystem::path path_;
    std::size_t size_;
    cuda::stream_ref stream_;
};

}  // namespace rapidsmpf
