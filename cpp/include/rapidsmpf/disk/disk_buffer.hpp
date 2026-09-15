/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <rapidsmpf/disk/disk_resource.hpp>

namespace rapidsmpf::disk {

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
     */
    explicit DiskBuffer(std::shared_ptr<DiskResource> disk);

    ~DiskBuffer();

    /// @brief Move constructor.
    /// @param other Buffer to move from.
    DiskBuffer(DiskBuffer&& other) noexcept;
    DiskBuffer& operator=(DiskBuffer&& other) = delete;  ///< Not move-assignable.
    DiskBuffer(DiskBuffer const&) = delete;  ///< Not copyable.
    DiskBuffer& operator=(DiskBuffer const&) = delete;  ///< Not copy-assignable.

    /**
     * @brief Current size of the backing file.
     *
     * @return Backing file size in bytes.
     */
    [[nodiscard]] std::uintmax_t file_size() const;

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
     * @brief Delete the backing file, if any.
     *
     * After deallocation `path()` is empty. Safe to call multiple times.
     */
    void deallocate() noexcept;

  private:
    std::shared_ptr<DiskResource> disk_;
    std::filesystem::path path_;
};

}  // namespace rapidsmpf::disk
