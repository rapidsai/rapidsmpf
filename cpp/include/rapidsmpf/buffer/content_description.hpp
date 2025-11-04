/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <numeric>
#include <type_traits>
#include <utility>

#include <rapidsmpf/buffer/buffer.hpp>

namespace rapidsmpf {

/**
 * @brief Description of an object's content.
 *
 * A `ContentDescription` encapsulates resource-related information about an
 * object's content such as memory sizes, memory types, and spillability.
 *
 * In RapidsMPF, an object's *content* refers to the actual data associated
 * with the object, not its metadata or auxiliary information. Typically, this
 * content is represented by one or more `Buffer` instances that may reside in
 * different memory spaces (e.g., host or device). It is also this content that
 * is subject to spilling and that typically accounts for the majority of an
 * object's overall memory footprint.
 *
 * The spillability state of an object is treated as an all-or-nothing property.
 * While one could imagine an object with a mix of spillable and non-spillable
 * content, this distinction is intentionally simplified in RapidsMPF.
 */
class ContentDescription {
  public:
    /** @brief Indicates whether the content is spillable. */
    enum class Spillable : bool {
        NO,
        YES
    };

    /**
     * @brief Construct a content description from a range of (MemoryType, size) pairs.
     *
     * Memory types omitted from the input are initialized to zero.
     *
     * @tparam Range A range whose value type is convertible to
     * `std::pair<MemoryType, std::size_t>`.
     * @param sizes Range of (MemoryType, size) pairs representing content sizes.
     * @param spillable Whether the content is spillable to slower memory tiers.
     *
     * @code{.cpp}
     * ContentDescription desc{
     *     std::array{
     *         std::pair{MemoryType::HOST,   1024UL},
     *         std::pair{MemoryType::DEVICE, 2048UL}
     *     },
     *     ContentDescription::Spillable::YES
     * };
     * @endcode
     */
    template <
        std::ranges::input_range Range =
            std::initializer_list<std::pair<MemoryType, std::size_t>>>
        requires std::convertible_to<
            std::ranges::range_value_t<Range>,
            std::pair<MemoryType, std::size_t>>
    constexpr explicit ContentDescription(Range&& sizes, Spillable spillable)
        : spillable_(spillable == Spillable::YES) {
        content_sizes_.fill(0);
        for (auto&& [mem_type, size] : sizes) {
            auto idx = static_cast<std::size_t>(mem_type);
            if (idx < content_sizes_.size()) {
                content_sizes_[idx] = size;
            }
        }
    }

    /**
     * @brief Construct a description with all sizes zero and a given spillability.
     *
     * Useful when you need a zero-sized content description or when building a content
     * description iteratively.
     *
     * @param spillable Whether the content are spillable.
     */
    constexpr ContentDescription(Spillable spillable = Spillable::NO)
        : ContentDescription({{}}, spillable) {}

    /**
     * @brief Access (read/write) the size for a specific memory type.
     *
     * @param mem_type The memory type entry to access.
     * @return Reference to the size (in bytes) for the given memory type.
     */
    [[nodiscard]] constexpr std::size_t& content_size(MemoryType mem_type) noexcept {
        return content_sizes_[static_cast<std::size_t>(mem_type)];
    }

    /**
     * @brief Get the size for a specific memory type.
     *
     * @param mem_type The memory type entry to access.
     * @return Size (in bytes) for the given memory type.
     */
    [[nodiscard]] constexpr std::size_t content_size(MemoryType mem_type) const noexcept {
        return content_sizes_[static_cast<std::size_t>(mem_type)];
    }

    /**
     * @brief Get the total content size across all memory types.
     *
     * Computes the sum of all per-memory-type content sizes.
     * This represents the total size (in bytes) of the object's content
     * across host, device, and any other memory types.
     *
     * @return Total size (in bytes) across all memory types.
     */
    [[nodiscard]] constexpr std::size_t content_size() const noexcept {
        return std::accumulate(
            content_sizes_.begin(), content_sizes_.end(), std::size_t{0}
        );
    }

    /// @brief @return Whether the content can be spilled.
    [[nodiscard]] constexpr bool spillable() const noexcept {
        return spillable_;
    }

    /**
     * @brief Equality comparison.
     *
     * @param other The content description to compare against.
     * @return `true` if both descriptions are equal; otherwise `false`.
     */
    [[nodiscard]] constexpr bool operator==(
        ContentDescription const& other
    ) const noexcept {
        return spillable_ == other.spillable_ && content_sizes_ == other.content_sizes_;
    }

  private:
    /// @brief Per memory-type content sizes, in bytes. Omitted memory types are
    /// initialized to zero-size entries.
    std::array<std::size_t, MEMORY_TYPES.size()> content_sizes_ = {};
    bool spillable_;
};

static_assert(
    std::is_trivially_copyable_v<ContentDescription>,
    "ContentDescription must be trivially copyable"
);

}  // namespace rapidsmpf
