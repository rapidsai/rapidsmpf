/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RAPIDSMPF_MEMORY_RESERVATION_RECORDER_H
#define RAPIDSMPF_MEMORY_RESERVATION_RECORDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define RAPIDSMPF_RECORDER_NOEXCEPT noexcept
#else
#define RAPIDSMPF_RECORDER_NOEXCEPT
#endif

#define RAPIDSMPF_RESERVATION_RECORDER_ABI_VERSION 1U

typedef enum {
    RAPIDSMPF_RESERVATION_ATTRIBUTE_STRING = 0,
    RAPIDSMPF_RESERVATION_ATTRIBUTE_I64 = 1,
    RAPIDSMPF_RESERVATION_ATTRIBUTE_BOOL = 2,
    RAPIDSMPF_RESERVATION_ATTRIBUTE_F64 = 3
} rapidsmpf_reservation_attribute_type;

/** @brief Borrowed UTF-8 string valid only for the duration of a callback. */
typedef struct {
    char const* data;  ///< String bytes; may be NULL only when size is zero.
    size_t size;  ///< Number of bytes, excluding any terminator.
} rapidsmpf_string_view;

/** @brief One borrowed reservation attribute. */
typedef struct {
    rapidsmpf_string_view key;  ///< Attribute key.
    uint32_t type;  ///< A rapidsmpf_reservation_attribute_type value.

    union {
        rapidsmpf_string_view string_value;  ///< String value.
        int64_t i64_value;  ///< Signed integer value.
        bool bool_value;  ///< Boolean value.
        double f64_value;  ///< Floating-point value.
    } value;  ///< Value selected by type.
} rapidsmpf_reservation_attribute;

/** @brief Borrowed attribute array valid only for the duration of a callback. */
typedef struct {
    rapidsmpf_reservation_attribute const* data;  ///< Attribute array.
    size_t size;  ///< Number of attributes.
} rapidsmpf_reservation_attributes_view;

typedef enum {
    RAPIDSMPF_MEMORY_DEVICE = 0,
    RAPIDSMPF_MEMORY_HOST = 1,
    RAPIDSMPF_MEMORY_PINNED_HOST = 2,
    RAPIDSMPF_MEMORY_DISK = 3
} rapidsmpf_recorder_memory_type;

typedef enum {
    RAPIDSMPF_REQUEST_IMMEDIATE = 0,
    RAPIDSMPF_REQUEST_OVERBOOK = 1,
    RAPIDSMPF_REQUEST_SPILL = 2,
    RAPIDSMPF_REQUEST_WAIT = 3,
    RAPIDSMPF_REQUEST_WAIT_OR_OVERBOOK = 4,
    RAPIDSMPF_REQUEST_WAIT_OR_FAIL = 5
} rapidsmpf_reservation_request_policy;

typedef enum {
    RAPIDSMPF_REQUEST_GRANTED = 0,
    RAPIDSMPF_REQUEST_EMPTY = 1,
    RAPIDSMPF_REQUEST_FAILED = 2,
    RAPIDSMPF_REQUEST_CANCELLED = 3
} rapidsmpf_reservation_request_outcome;

typedef enum {
    RAPIDSMPF_RESERVATION_CONSUMED = 0,
    RAPIDSMPF_RESERVATION_SPLIT_OFF = 1
} rapidsmpf_reservation_resize_reason;

typedef enum {
    RAPIDSMPF_RESERVATION_CLEARED = 0,
    RAPIDSMPF_RESERVATION_RELEASED_CONSUMED = 1,
    RAPIDSMPF_RESERVATION_RELEASED_SPLIT = 2
} rapidsmpf_reservation_release_reason;

/** @brief Versioned callback table for observing reservation activity. */
typedef struct rapidsmpf_reservation_recorder {
    uint32_t abi_version;  ///< RAPIDSMPF_RESERVATION_RECORDER_ABI_VERSION.
    size_t struct_size;  ///< Size of this structure in bytes.
    void* user;  ///< Opaque callback state.
    void (*retain_user)(void*) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Retain user state.
    void (*release_user)(void*) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Release user state.
    void (*request_submitted)(
        void*,
        uint64_t,
        uint64_t,
        size_t,
        uint32_t const*,
        size_t,
        uint32_t,
        bool,
        int64_t,
        rapidsmpf_reservation_attributes_view
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe request submission.
    void (*request_waiting)(
        void*, uint64_t, uint64_t
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe entry into the wait queue.
    void (*request_resolved)(
        void*,
        uint64_t,
        uint64_t,
        uint32_t,
        uint64_t,
        uint32_t,
        size_t,
        size_t,
        bool,
        rapidsmpf_string_view
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe a terminal request outcome.
    void (*reservation_created)(
        void*,
        uint64_t,
        uint64_t,
        uint32_t,
        size_t,
        rapidsmpf_reservation_attributes_view,
        uint64_t,
        uint64_t
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe reservation creation.
    void (*reservation_resized)(
        void*, uint64_t, uint64_t, size_t, uint32_t
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe a nonzero reservation resize.
    void (*reservation_released)(
        void*, uint64_t, uint64_t, uint32_t
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe reservation release.
    void (*memory_limit_changed)(
        void*, uint64_t, uint32_t, int64_t
    ) RAPIDSMPF_RECORDER_NOEXCEPT;  ///< Observe an exact signed memory limit.
} rapidsmpf_reservation_recorder;

#ifdef __cplusplus
}
#undef RAPIDSMPF_RECORDER_NOEXCEPT
#endif

#endif
