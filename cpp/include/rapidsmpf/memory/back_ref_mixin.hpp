/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <utility>

namespace rapidsmpf {

/**
 * @brief Mixin that contributes a weak back-reference to an owner and promotes
 * it to shared ownership on copy.
 *
 * Inherit this mixin into a type whose copies should keep an external owner
 * alive. The mixin holds a `std::weak_ptr<BackRef>` plus an internal
 * `std::shared_ptr<BackRef>` that is populated by the copy constructor and
 * copy assignment operator.
 *
 * The mixin is the building block previously provided by the now-removed
 * `OwningResourceAdaptor`: it captures the "back-reference acquires shared
 * ownership when CCCL deep-copies a resource into an owning
 * `cuda::mr::any_resource`" lifetime trick without the surrounding
 * allocate/deallocate/`forward_property` wrapper.
 *
 * @par Lifetime semantics
 *
 * - A default-constructed mixin holds an empty `weak_` and empty `strong_`.
 *   Copying such an instance is a no-op for the back-ref (no throw, no
 *   promotion). This is the "back-ref not yet installed" state.
 * - `set_backref()` installs a `std::weak_ptr<BackRef>` into the mixin and
 *   clears any previously promoted strong reference. After this call, copies
 *   of the host object will attempt to promote the weak reference.
 * - On copy, if the source already holds a `strong_` reference, the copy
 *   shares it. Otherwise the copy promotes `weak_` to a `shared_ptr`. If
 *   `weak_` was installed but has since expired, the promotion throws
 *   `std::bad_weak_ptr`, matching the behavior the previous
 *   `OwningResourceAdaptor` used to detect dangling-copy scenarios early.
 * - Moves are trivial: the moved-from instance leaves its pointers in their
 *   moved-from state (typically empty), and the moved-to instance takes over
 *   any installed weak/strong references without re-promotion.
 *
 * @tparam BackRef Type of the back-referenced owner object.
 */
template <typename BackRef>
class BackRefMixin {
  public:
    /// @brief Construct a mixin with no installed back-reference.
    BackRefMixin() noexcept = default;

    /**
     * @brief Copy constructor that promotes the back-reference to shared
     * ownership.
     *
     * - If @p other already holds a strong back-reference, it is shared.
     * - Otherwise, if @p other has an installed weak back-reference, it is
     *   promoted via `std::shared_ptr<BackRef>{other.weak_}`.
     * - If @p other has no installed back-reference (default state), this is
     *   a no-op for the back-ref state.
     *
     * @param other Mixin to copy from.
     * @throws std::bad_weak_ptr if @p other has an installed weak
     * back-reference that has since expired.
     */
    BackRefMixin(BackRefMixin const& other)
        : weak_{other.weak_}, strong_{promote(other)} {}

    /**
     * @brief Copy assignment operator with the same lifetime semantics as the
     * copy constructor.
     *
     * @param other Mixin to copy from.
     * @return Reference to this mixin.
     * @throws std::bad_weak_ptr if @p other has an installed weak
     * back-reference that has since expired.
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
     * @brief Owner-based equality of the back-reference.
     *
     * Two mixins compare equal when their weak references refer to the same
     * control block under `std::weak_ptr::owner_before` (or when both are
     * empty). The strong reference does not participate: a mixin that has
     * promoted its weak ref via copy compares equal to its source.
     *
     * This replicates the back-ref half of the equality previously exposed by
     * `OwningResourceAdaptor::operator==`. Combine with the host type's own
     * identity check to get full equality.
     *
     * @param other Mixin to compare against.
     * @return `true` if both mixins reference the same back-referenced
     * owner (or both are uninstalled).
     */
    [[nodiscard]] bool operator==(BackRefMixin const& other) const noexcept {
        return !weak_.owner_before(other.weak_) && !other.weak_.owner_before(weak_);
    }

    /**
     * @brief Install a weak back-reference.
     *
     * Clears any previously promoted strong reference so a subsequent copy
     * will re-promote from the new weak reference.
     *
     * @param backref Weak reference to the back-referenced owner object.
     */
    void set_backref(std::weak_ptr<BackRef> backref) noexcept {
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
        // "installed but expired". An installed weak_ptr has a non-empty
        // owner; default-constructed ones owner-compare equal to each other.
        std::weak_ptr<BackRef> const empty{};
        bool const installed =
            other.weak_.owner_before(empty) || empty.owner_before(other.weak_);
        if (!installed) {
            return {};
        }
        return std::shared_ptr<BackRef>{other.weak_};  // throws if expired
    }

    std::weak_ptr<BackRef> weak_{};
    std::shared_ptr<BackRef> strong_{};
};

}  // namespace rapidsmpf
