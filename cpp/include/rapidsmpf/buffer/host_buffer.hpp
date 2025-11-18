/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @brief A host buffer that wraps a vector of bytes.
 *
 * This class is used to wrap a vector of bytes and provide a view into its data.
 * The buffer does not allocate its own memory but instead references
 * the vector's internal storage.
 */
class HostBuffer {
  public:
    HostBuffer() = default;

    /**
     * @brief Constructs a host buffer with the specified size.
     *
     * @param size The size of the buffer in bytes.
     */
    constexpr HostBuffer(size_t size)
        : size_(size), data_(size > 0 ? allocator_.allocate(size) : nullptr) {}

    /**
     * @brief Constructs a host buffer and copies data into it.
     *
     * @param src_data Pointer to the source data to copy.
     * @param size The size of the data to copy in bytes.
     */
    HostBuffer(void const* src_data, size_t size) : HostBuffer(size) {
        std::memcpy(data_, src_data, size);
    }

    /**
     * @brief Constructs a host buffer that wraps an existing vector.
     *
     * Takes ownership of the vector and provides a view into its data.
     * The buffer does not allocate its own memory but instead references
     * the vector's internal storage.
     *
     * @param data A unique pointer to the vector to wrap. Must not be nullptr.
     *
     * @throws std::invalid_argument If @p data is nullptr.
     */
    explicit HostBuffer(std::unique_ptr<std::vector<uint8_t>> data) {
        RAPIDSMPF_EXPECTS(
            data != nullptr, "the data pointer cannot be nullptr", std::invalid_argument
        );
        size_ = data->size();
        data_ = reinterpret_cast<std::byte*>(data->data());
        parent_buf_ = std::move(data);
    }

    /**
     * @brief Destroys the host buffer and frees allocated memory.
     *
     * If this buffer wraps a vector (via parent_buf_), the vector is destroyed.
     * Otherwise, if this buffer owns its own allocation, the memory is deallocated.
     */
    ~HostBuffer() noexcept {
        clear();
    }

    /**
     * @brief Clears the host buffer and frees allocated memory.
     */
    void clear() noexcept {
        if (parent_buf_) {
            parent_buf_.reset();
        } else if (data_ != nullptr && size_ > 0) {
            allocator_.deallocate(data_, size_);
        }
        data_ = nullptr;
        size_ = 0;
    }

    /**
     * @brief Move constructor.
     *
     * @param other The host buffer to move from.
     */
    HostBuffer(HostBuffer&& other) noexcept
        : allocator_(std::move(other.allocator_)),
          size_(other.size_),
          data_(other.data_),
          parent_buf_(std::move(other.parent_buf_)) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator.
     *
     * @param other The host buffer to move from.
     * @return Reference to this object.
     */
    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            clear();

            // Move from other
            allocator_ = std::move(other.allocator_);
            size_ = other.size_;
            data_ = other.data_;
            parent_buf_ = std::move(other.parent_buf_);

            // Reset other (parent_buf_ is already null after move)
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Copy constructor.

     * @param other The host buffer to copy from.
     */
    HostBuffer(const HostBuffer& other) : HostBuffer(other.data(), other.size()) {}

    /**
     * @brief Copy assignment operator.
     *
     * @param other The host buffer to copy from.
     * @return Reference to this object.
     */
    HostBuffer& operator=(const HostBuffer& other) {
        if (this != &other) {
            HostBuffer temp(other);
            *this = std::move(temp);
        }
        return *this;
    }

    /**
     * @brief Gets a const pointer to the buffer data.
     *
     * @return A const pointer to the buffer data, or nullptr if not allocated.
     */
    [[nodiscard]] constexpr std::byte const* data() const noexcept {
        return data_;
    }

    /**
     * @brief Gets a pointer to the buffer data.
     *
     * @return A pointer to the buffer data, or nullptr if not allocated.
     */
    constexpr std::byte* data() noexcept {
        return data_;
    }

    /**
     * @brief Gets the size of the buffer.
     *
     * @return The size of the buffer in bytes.
     */
    [[nodiscard]] constexpr size_t size() const noexcept {
        return size_;
    }

  private:
    std::allocator<std::byte> allocator_{};
    size_t size_{0};  ///< Size of the buffer in bytes.
    std::byte* data_{nullptr};  ///< Pointer to the allocated buffer data.
    std::unique_ptr<std::vector<uint8_t>> parent_buf_{
        nullptr
    };  ///< Pointer to the parent vector, if this was constructed from a vector.
};
}  // namespace rapidsmpf
