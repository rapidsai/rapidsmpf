/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <utility>

#include <cuda/stream>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>

namespace rapidsmpf {

class BufferResource;
class DiskResource;

/**
 * @brief Opaque handle to a file managed by a `DiskResource`.
 *
 * A `FileHandle` can only be created by the `DiskResource`, and all file operations
 * (creation, I/O, size queries, deletion and adoption of external files) go through that
 * resource. A default-constructed `FileHandle` refers to no file.
 */
class FileHandle {
  public:
    /// @brief Create a handle that refers to no file.
    FileHandle() = default;

    /**
     * @brief Whether this handle refers to no file.
     *
     * @return `true` if the handle is empty.
     */
    [[nodiscard]] bool empty() const noexcept {
        return path_.empty();
    }

    /**
     * @brief Current size of the file.
     *
     * @return File size in bytes.
     * @throws std::logic_error if the handle is empty.
     * @throws std::filesystem::filesystem_error if the size cannot be queried, e.g. the
     *         file was removed.
     * @throws std::overflow_error if the size does not fit in `std::size_t`.
     */
    [[nodiscard]] std::size_t file_size() const;

    /**
     * @brief Compare two handles.
     *
     * @param other Handle to compare with.
     * @return `true` if both refer to the same path.
     */
    [[nodiscard]] bool operator==(FileHandle const& other) const noexcept = default;

  private:
    friend class DiskResource;

    explicit FileHandle(std::filesystem::path path) : path_{std::move(path)} {}

    std::filesystem::path path_;
};

/**
 * @brief Disk I/O for host or device byte buffers.
 *
 * Uses KvikIO with CompatMode::AUTO (GDS when available, POSIX/compat otherwise).
 *
 * Each resource atomically creates and exclusively owns its directory. The directory
 * is removed when the resource is destroyed.
 */
class DiskResource : public BackRefMixin<BufferResource> {
  public:
    ~DiskResource() noexcept;

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
     * @return Handle to the reserved empty file.
     */
    [[nodiscard]] FileHandle create_unique_path() const;

    /**
     * @brief Write bytes to a file and block until the transfer completes.
     *
     * @param path File.
     * @param data Host or device pointer to the source bytes.
     * @param size Number of bytes to write.
     * @param stream CUDA stream associated with @p data.
     * @param file_offset Byte offset within the file. Existing bytes outside
     *        the written range are preserved when the file already exists.
     *        A missing file is created.
     * @return Number of bytes transferred. The caller must check this against
     *         @p size.
     */
    [[nodiscard]] std::size_t write(
        FileHandle const& path,
        void const* data,
        std::size_t size,
        cuda::stream_ref stream,
        std::ptrdiff_t file_offset = 0
    ) const;

    /**
     * @brief Read bytes from a file and block until the transfer completes.
     *
     * @param path File.
     * @param data Host or device pointer to the destination buffer. Must remain
     *        valid until this call returns.
     * @param size Number of bytes to read.
     * @param stream CUDA stream associated with @p data.
     * @param file_offset Byte offset within the file.
     * @return Number of bytes transferred. The caller must check this against
     *         @p size.
     */
    [[nodiscard]] std::size_t read(
        FileHandle const& path,
        void* data,
        std::size_t size,
        cuda::stream_ref stream,
        std::ptrdiff_t file_offset = 0
    ) const;

    /**
     * @brief Copy bytes from one file to another and block until the transfer completes.
     *
     * Synchronizes @p stream first, so the copy runs after all work already enqueued on
     * it (as in `write()` and `read()`). The transfer itself is host-side, using
     * `copy_file_range(2)`. The destination is created if missing, and existing bytes
     * outside the written range are preserved.
     *
     * @param src_path Source file.
     * @param dst_path Destination file. Must differ from @p src_path.
     * @param size Number of bytes to copy.
     * @param stream CUDA stream to synchronize before the copy.
     * @param src_offset Byte offset within the source file.
     * @param dst_offset Byte offset within the destination file.
     * @return Number of bytes transferred. The caller must check this against
     *         @p size.
     * @throws std::invalid_argument if @p src_path equals @p dst_path or an offset is
     *         negative.
     * @throws std::runtime_error if the kernel or filesystem rejects the copy.
     */
    [[nodiscard]] std::size_t copy(
        FileHandle const& src_path,
        FileHandle const& dst_path,
        std::size_t size,
        cuda::stream_ref stream,
        std::ptrdiff_t src_offset = 0,
        std::ptrdiff_t dst_offset = 0
    ) const;

    /**
     * @brief Take ownership of an external file by placing it under `directory()`.
     *
     * If @p path is already directly under `directory()` (compared by device and inode,
     * so equivalent spellings of the directory match), it is wrapped unchanged.
     * Otherwise the file is relocated to a new unique path under `directory()`.
     * This uses `rename(2)` (atomic, no copy) and, if @p path is on a different
     * filesystem, falls back to copying the file and removing the source, which is
     * proportional to the file size. If relocation fails, the source is left in place
     * and nothing is left behind in `directory()`.
     *
     * @param path Existing regular file to adopt.
     * @return Handle to the file under `directory()`.
     * @throws std::invalid_argument if @p path is not an existing regular file.
     * @throws std::runtime_error if the file cannot be relocated.
     */
    [[nodiscard]] FileHandle adopt(std::filesystem::path const& path) const;

    /**
     * @brief Take ownership of a file managed by another `DiskResource`.
     *
     * Same as the overload taking a filesystem path: the file is returned unchanged if it
     * is already under `directory()`, and relocated otherwise.
     *
     * @param path Handle to the file to adopt.
     * @return Handle to the file under `directory()`.
     * @throws std::invalid_argument if @p path is empty or not an existing regular file.
     * @throws std::runtime_error if the file cannot be relocated.
     */
    [[nodiscard]] FileHandle adopt(FileHandle const& path) const;

    /**
     * @brief Delete a file and reset the handle.
     *
     * Does nothing if @p path is empty. Failures are logged, not thrown, so this is safe
     * to call from destructors. The handle is empty afterwards, even on failure.
     *
     * @param path Handle to the file to delete.
     * @return `true` if the file was deleted or @p path was empty, `false` if the
     *         deletion failed.
     */
    bool remove(FileHandle& path) const noexcept;

    /**
     * @brief Durably synchronize file data to storage.
     *
     * Not used on the default spill path; exposed for benchmark durability cases.
     *
     * @param path File.
     */
    void flush(FileHandle const& path) const;

    /**
     * @brief Compare two disk resources.
     *
     * @param other Resource to compare with.
     * @return `true` if both have the same directory and back-reference state.
     */
    [[nodiscard]] bool operator==(DiskResource const& other) const noexcept = default;

  private:
    explicit DiskResource(std::filesystem::path const& parent);

    friend class BufferResource;

    std::filesystem::path dir_;
};

/**
 * @brief Spill directory from options.
 *
 * Disabled values (`false`, `none`, …) yield `std::nullopt`. An empty
 * string is treated as unset and uses the default (`false`). A
 * whitespace-only value is rejected.
 *
 * @param options Configuration options.
 * @return Configured directory, if disk spilling is enabled.
 * @throws std::invalid_argument if the option is whitespace-only.
 */
[[nodiscard]] std::optional<std::filesystem::path> spill_dir_from_options(
    config::Options options
);

}  // namespace rapidsmpf
