/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief Mixin that lets copies of the this object keep an external object reference of
 * type @p BackRef alive.
 *
 * @note Copying an instance without calling `set_backref()` throws `std::bad_weak_ptr`.
 *
 * @tparam BackRef Type of the external object reference.
 */
template <typename BackRef>
class BackRefMixin {
  public:
    /// @brief Construct a mixin with no installed back-reference.
    BackRefMixin() noexcept = default;

    /**
     * @brief Acquire shared ownership of @p other's back-referenced owner.
     *
     * After construction, *this* references the same owner as @p other and
     * keeps it alive for as long as *this* lives.
     *
     * @param other Mixin to copy from.
     * @throws std::bad_weak_ptr if @p other is uninstalled, or is installed
     * but its owner has been destroyed.
     */
    BackRefMixin(BackRefMixin const& other)
        : weak_{other.weak_},
          strong_{other.strong_ ? other.strong_ : std::shared_ptr<BackRef>{other.weak_}} {
    }

    /**
     * @brief Copy assignment with the same semantics as the copy constructor.
     *
     * If the assignment throws, *this* is left unchanged.
     *
     * @param other Mixin to copy from.
     * @return Reference to this.
     * @throws std::bad_weak_ptr if @p other is uninstalled, or is installed
     * but its owner has been destroyed.
     */
    BackRefMixin& operator=(BackRefMixin const& other) {
        if (this != &other) {
            // Promote first so a throw leaves *this unchanged.
            auto promoted =
                other.strong_ ? other.strong_ : std::shared_ptr<BackRef>{other.weak_};
            weak_ = other.weak_;
            strong_ = std::move(promoted);
        }
        return *this;
    }

    /// @brief Move constructor.
    BackRefMixin(BackRefMixin&&) noexcept = default;

    /**
     * @brief Move assignment operator.
     * @return Reference to this mixin.
     */
    BackRefMixin& operator=(BackRefMixin&&) noexcept = default;

    ~BackRefMixin() = default;

    /**
     * @brief Owner-based equality.
     *
     * @param other Mixin to compare against.
     * @return `true` if both mixins reference the same back-referenced
     * owner, or are both uninstalled.
     */
    [[nodiscard]] bool operator==(BackRefMixin const& other) const noexcept {
        return owner_equal(weak_, other.weak_);
    }

    /**
     * @brief Install a back-reference on this instance.
     *
     * After this call, every subsequent copy of *this* will hold shared
     * ownership of @p backref's referent for the lifetime of the copy.
     *
     * Installation only affects *this* and copies made from it afterwards;
     * copies that already exist are not retroactively back-referenced. To
     * guarantee the back-reference is honored by every observable copy,
     * install it before the host object becomes reachable to any code that
     * may copy it.
     *
     * @param backref Non-empty weak reference to the back-referenced owner.
     * @throws std::invalid_argument if @p backref is empty, or if a
     * back-reference is already installed on this instance.
     */
    void set_backref(std::weak_ptr<BackRef> backref) {
        std::weak_ptr<BackRef> const empty_weak{};
        RAPIDSMPF_EXPECTS(
            !owner_equal(backref, empty_weak),
            "set_backref: backref must not be empty",
            std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            owner_equal(weak_, empty_weak),
            "set_backref: backref is already set",
            std::invalid_argument
        );
        weak_ = std::move(backref);
        strong_.reset();
    }

  private:
    std::weak_ptr<BackRef> weak_{};
    std::shared_ptr<BackRef> strong_{};
};

}  // namespace rapidsmpf
