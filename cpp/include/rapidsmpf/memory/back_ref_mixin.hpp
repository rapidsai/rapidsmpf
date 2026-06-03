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
 * - **Uninstalled** (default-constructed): the instance makes no claim on
 *   any owner. Copies of it stay uninstalled and never throw.
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
     * keeps it alive for as long as *this* lives. If @p other is
     * uninstalled, *this* is uninstalled too.
     *
     * @param other Mixin to copy from.
     * @throws std::bad_weak_ptr if @p other is installed but its owner has
     * been destroyed.
     */
    BackRefMixin(BackRefMixin const& other)
        : weak_{other.weak_}, strong_{promote(other)} {}

    /**
     * @brief Copy assignment with the same semantics as the copy constructor.
     *
     * If the assignment throws, *this* is left unchanged.
     *
     * @param other Mixin to copy from.
     * @return Reference to this.
     * @throws std::bad_weak_ptr if @p other is installed but its owner has
     * been destroyed.
     */
    BackRefMixin& operator=(BackRefMixin const& other) {
        if (this != &other) {
            // Promote first so a throw leaves *this unchanged.
            auto promoted = promote(other);
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
    /**
     * @brief Compute the promoted strong reference for a copy from @p other.
     *
     * Returns an empty `shared_ptr` when @p other has no installed
     * back-reference (so default-constructed mixins copy without throwing).
     */
    static std::shared_ptr<BackRef> promote(BackRefMixin const& other) {
        if (other.strong_) {
            return other.strong_;
        }
        // Distinguish "never installed" (default-constructed weak_ptr) from
        // "installed but expired": only the former is owner-equal to a
        // default-constructed weak_ptr.
        if (owner_equal(other.weak_, std::weak_ptr<BackRef>{})) {
            return {};
        }
        return std::shared_ptr<BackRef>{other.weak_};  // throws if expired
    }

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
