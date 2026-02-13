/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rapidsmpf {

/**
 * @brief Buffer of fixed-size host memory blocks with type-erased storage.
 *
 * Holds a total size in bytes, a block size, and a span of block start pointers.
 * Storage is type-erased via `unique_ptr<void, deleter>`, so different backends
 * can be used: a single vector (split into blocks), a vector of vectors, or
 * e.g. cucascade's multiple_blocks_allocation.
 *
 * Example wrapping multiple_blocks_allocation (via a factory or friend that
 * calls the private constructor):
 * @code
 *   auto alloc = multiple_blocks_allocation::create(blocks, mr);
 *   auto blocks_span = alloc->get_blocks();
 *   FixedSizedHostBuffer buf(alloc->size_bytes(), alloc->block_size(), blocks_span,
 *       alloc.get(), [a = std::move(alloc)](void*) mutable { a.reset(); });
 * @endcode
 */
class FixedSizedHostBuffer {
  public:
    /// Type-erased deleter invoked with the storage pointer on destruction.
    using storage_deleter_type = std::function<void(void*)>;

    /// @brief Default block size of 1 MiB.
    static constexpr size_t default_block_size = size_t(1) << 20;

    /**
     * @brief Construct an empty buffer with a given block size.
     * @param block_size Size of each block in bytes.
     */
    explicit FixedSizedHostBuffer(size_t block_size = default_block_size)
        : block_size_(block_size) {}

    /**
     * @brief Construct from a single contiguous vector split into fixed-size blocks.
     *
     * Takes ownership of @p vec by moving it into internal storage.
     *
     * @param vec Contiguous bytes (moved from).
     * @param block_size Size of each block in bytes.
     * @return A buffer with blocks covering the vector.
     */
    static FixedSizedHostBuffer from_vector(
        std::vector<std::byte> vec, std::size_t block_size
    );

    /**
     * @brief Construct from a vector of vectors (one block per inner vector).
     *
     * Takes ownership of @p vecs. Each inner vector becomes one block; all must
     * have the same size.
     *
     * @param vecs Vector of byte vectors (moved from).
     * @return A buffer with one block per inner vector.
     */
    static FixedSizedHostBuffer from_vectors(std::vector<std::vector<std::byte>> vecs);

    FixedSizedHostBuffer(FixedSizedHostBuffer const&) = delete;
    FixedSizedHostBuffer& operator=(FixedSizedHostBuffer const&) = delete;

    /**
     * @brief Move constructor; the moved-from buffer is left empty.
     * @param other Buffer to move from.
     */
    FixedSizedHostBuffer(FixedSizedHostBuffer&& other) noexcept;

    /**
     * @brief Move assignment; the moved-from buffer is left empty.
     * @param other Buffer to move from.
     * @return Reference to this buffer.
     */
    FixedSizedHostBuffer& operator=(FixedSizedHostBuffer&& other) noexcept;

    /**
     * @brief Total size in bytes across all blocks.
     * @return Total number of bytes.
     */
    [[nodiscard]] constexpr std::size_t total_size() const noexcept {
        return total_size_;
    }

    /**
     * @brief Size of each block in bytes.
     * @return Block size in bytes.
     */
    [[nodiscard]] constexpr std::size_t block_size() const noexcept {
        return block_size_;
    }

    /**
     * @brief Number of blocks.
     * @return Number of blocks.
     */
    [[nodiscard]] constexpr std::size_t num_blocks() const noexcept {
        return block_ptrs_.size();
    }

    /**
     * @brief Span of block start pointers (mutable).
     * @return Span of block start pointers.
     */
    [[nodiscard]] constexpr std::span<std::byte*> blocks() noexcept {
        return block_ptrs_;
    }

    /**
     * @brief Span of block start pointers (const).
     * @return Span of block start pointers.
     */
    [[nodiscard]] constexpr std::span<std::byte* const> blocks() const noexcept {
        return block_ptrs_;
    }

    /**
     * @brief True if there are no blocks.
     * @return True if empty, false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return block_ptrs_.empty();
    }

    /**
     * @brief Reset to empty state (release storage, zero sizes, clear block span).
     */
    void reset() noexcept;

    /**
     * @brief The i-th block as a span of bytes.
     *
     * @param i Block index in [0, num_blocks()).
     * @return Span of length block_size() over the block's bytes.
     * @throws std::out_of_range if i >= num_blocks().
     */
    [[nodiscard]] std::span<std::byte> block_data(std::size_t i);

    /**
     * @brief The i-th block as a span of bytes.
     *
     * @param i Block index in [0, num_blocks()).
     * @return Span of length block_size() over the block's bytes.
     * @throws std::out_of_range if i >= num_blocks().
     */
    [[nodiscard]] std::span<std::byte const> block_data(std::size_t i) const;

    /**
     * @brief Type-erased constructor: take ownership of storage and block metadata.
     *
     * The deleter is invoked with the storage pointer when this buffer is destroyed.
     * @p block_ptrs must refer to memory that remains valid for the lifetime of this
     * buffer (typically inside the storage), e.g. from get_blocks() on
     * multiple_blocks_allocation.
     *
     * @param size Total size in bytes.
     * @param block_size Size of each block in bytes.
     * @param block_ptrs View of block start pointers (not copied; must outlive this
     * buffer).
     * @param storage Type-erased pointer to the storage (e.g. vector, allocation
     * wrapper).
     * @param deleter Called with @p storage on destruction.
     */
    FixedSizedHostBuffer(
        std::size_t size,
        std::size_t block_size,
        std::span<std::byte*> block_ptrs,
        void* storage,
        storage_deleter_type deleter
    )
        : storage_(storage, std::move(deleter)),
          total_size_(size),
          block_size_(block_size),
          block_ptrs_(block_ptrs) {}

  private:
    std::unique_ptr<void, storage_deleter_type> storage_{};
    std::size_t total_size_{0};
    std::size_t block_size_{default_block_size};
    std::span<std::byte*> block_ptrs_{};
};

}  // namespace rapidsmpf
