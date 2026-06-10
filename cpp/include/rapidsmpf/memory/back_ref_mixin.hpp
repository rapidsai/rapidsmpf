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
 * @brief Mixin that lets copies of the host object keep an external owner alive.
 *
 * Inherit this mixin into a type whose copies should share ownership of an
 * external @p BackRef instance. Each instance is in one of two states:
 *
 * - **Uninstalled** (default-constructed): the instance is not bound to any
 *   owner. Copying an uninstalled instance throws `std::bad_weak_ptr`; a
 *   back-reference must be installed via `set_backref()` before copying.
 * - **Installed** (after `set_backref()`): the instance is bound to a
 *   specific owner. Each copy of an installed instance acquires shared
 *   ownership of that owner for the lifetime of the copy. If the owner has
 *   been destroyed before the copy is made, copying throws
 *   `std::bad_weak_ptr`.
 *
 * Move operations transfer state without re-acquiring ownership. Equality
 * is owner-based: two instances compare equal iff they reference the same
 * owner, or are both uninstalled.
 *
 * @tparam BackRef Type of the back-referenced owner object.
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
        // Reuse `other`'s strong ref when present; otherwise lock its weak ref,
        // which throws `std::bad_weak_ptr` if `other` is uninstalled or expired.
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
            // Promote first so a throw leaves *this unchanged. Locking an
            // uninstalled or expired weak ref throws `std::bad_weak_ptr`.
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
     * @throws std::invalid_argument if @p backref is empty.
     */
    void set_backref(std::weak_ptr<BackRef> backref) {
        RAPIDSMPF_EXPECTS(
            !owner_equal(backref, std::weak_ptr<BackRef>{}),
            "set_backref: backref must not be empty",
            std::invalid_argument
        );
        weak_ = std::move(backref);
        strong_.reset();
    }

  private:
    std::weak_ptr<BackRef> weak_{};
    std::shared_ptr<BackRef> strong_{};
};

class BufferResource;

/**
 * @brief Convenience alias: `BackRefMixin` instantiated for `BufferResource`.
 *
 * Inherit from this alias to give a type the back-reference lifetime
 * contract that `BufferResource::create()` installs on its internal
 * resources. See `BackRefMixin` for the contract.
 */
using WithBufferResourceBackRef = BackRefMixin<BufferResource>;

}  // namespace rapidsmpf
