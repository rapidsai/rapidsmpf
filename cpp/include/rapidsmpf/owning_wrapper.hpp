/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace rapidsmpf {

/**
 * @brief Utility class to store an arbitrary type-erased object while another object is
 * alive.
 *
 * When sending messages through `Channel`s from Python, we typically need to keep various
 * Python objects alive since the matching C++ objects only hold views.
 *
 * For example, when constructing a `TableChunk` from a pylibcudf `Table`, the
 * `TableChunk` has a non-owning `cudf::table_view` of the `Table` and someone must be
 * responsible for keeping the `Table` alive for the lifetime of the `TableChunk`. If we
 * want to allow creation of such objects in Python with the ability to sink them on the
 * C++ side we cannot rely on the Python side of things keeping the `Table` alive (the
 * reference disappears!). Similarly when we send a message through a `Channel` the sender
 * will, once pushed into the channel, drop the reference to the message payload and so,
 * again, we need some way of keeping the payload alive.
 *
 * To square this circle, such C++ objects have an `OwningWrapper` slot that stores a
 * type-erased pointer with, as far as we are concerned, unique ownership semantics. When
 * this object is destroyed, the custom deleter runs and can do whatever deallocation is
 * necessary.
 *
 * @warning Behaviour is undefined if the unique ownership semantic is not respected. The
 * deleter may be called from any thread at any time, the implementer of the deleter is
 * responsible for correct synchronisation with (for example) the Python GIL. Furthermore,
 * the deleter may not throw: if an error occurs, the only safe thing to do is
 * `std::terminate`.
 *
 * @warning When using this `OwningWrapper` inside a C++ object, make sure it is
 * constructed first and destructed last.
 */
class OwningWrapper {
  public:
    /// @brief Callback used to delete the owned object.
    using deleter_type = void (*)(void*);

    OwningWrapper() = default;

    /**
     * @brief Take ownership and responsibility for the destruction of an object.
     *
     * @param obj Type-erased object to own.
     * @param deleter Function called to destruct the object.
     */
    explicit OwningWrapper(void* obj, deleter_type deleter)
        : obj_{owning_type(obj, deleter)} {}

    /**
     * @brief Release ownership of the underlying pointer
     *
     * @return Pointer to object.
     */
    [[nodiscard]] void* release() noexcept {
        return obj_.release();
    }

    /**
     * @brief @return Get access to the underlying pointer.
     */
    [[nodiscard]] void* get() const noexcept {
        return obj_.get();
    }

  private:
    using owning_type = std::unique_ptr<void, deleter_type>;
    owning_type obj_{nullptr, [](void*) {}};
};
}  // namespace rapidsmpf
