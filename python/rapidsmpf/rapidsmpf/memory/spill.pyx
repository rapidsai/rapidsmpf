# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spill and unspill packed partitions between device and host memory."""

from cython.operator cimport dereference as deref
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer_resource cimport (AllowOverbooking,
                                               BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)


cdef vector[cpp_PackedData] _partitions_py_to_cpp(partitions):
    cdef vector[cpp_PackedData] ret
    for part in partitions:
        if not (<PackedData?>part).c_obj:
            raise ValueError("PackedData was empty")
        ret.push_back(move(deref((<PackedData?>part).c_obj)))
    return move(ret)


cdef extern from "<rapidsmpf/memory/spill.hpp>" nogil:
    cdef vector[cpp_PackedData] cpp_spill_partitions \
        "rapidsmpf::spill_partitions"(
            vector[cpp_PackedData] partitions,
            cpp_BufferResource* br,
        ) except +ex_handler

    cdef vector[cpp_PackedData] cpp_unspill_partitions \
        "rapidsmpf::unspill_partitions"(
            vector[cpp_PackedData] partitions,
            cpp_BufferResource* br,
            AllowOverbooking allow_overbooking,
        ) except +ex_handler


cpdef object spill_partitions(
    object partitions,
    BufferResource br,
):
    """
    Spill partitions from device memory to host memory.

    Moves the buffer of each ``PackedData`` from device memory to host memory using
    the provided buffer resource and the buffer's CUDA stream. Partitions already
    in host memory are returned unchanged.

    For device-resident partitions, a host memory reservation is made before moving
    the buffer. If the reservation fails due to insufficient host memory, an
    exception is raised. Overbooking is not allowed.

    The input partitions are released and are left empty on return.

    Parameters
    ----------
    partitions
        The partitions to spill.
    br
        Buffer resource used to reserve host memory and perform the move.

    Returns
    -------
    A list of partitions whose buffers reside in host memory.

    Raises
    ------
    ReservationError
        If host memory reservation fails.
    """
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef vector[cpp_PackedData] _ret
    with nogil:
        _ret = cpp_spill_partitions(
            move(_partitions),
            _br,
        )
    return packed_data_vector_to_list(move(_ret), br)


cpdef object unspill_partitions(
    object partitions,
    BufferResource br,
    object allow_overbooking,
):
    """
    Move spilled partitions back to device memory.

    Each partition is inspected to determine whether its buffer resides in device
    memory. Buffers already in device memory are left untouched. Host-resident buffers
    are moved to device memory using the provided buffer resource and the buffer's CUDA
    stream.

    If insufficient device memory is available, the buffer resource's spill manager is
    invoked to free memory. If overbooking occurs and spilling fails to reclaim enough
    memory, behavior depends on ``allow_overbooking``.

    The input partitions are released and are left empty on return.

    Parameters
    ----------
    partitions
        The partitions to unspill, potentially containing host-resident data.
    br
        Buffer resource responsible for memory reservation and spills.
    allow_overbooking
        If False, ensures enough memory is freed to satisfy the reservation;
        otherwise, allows overbooking even if spilling was insufficient.

    Returns
    -------
    A list of partitions whose buffers reside in device memory.

    Raises
    ------
    ReservationError
        If overbooking exceeds the amount spilled and ``allow_overbooking is False``.
    """
    if not isinstance(allow_overbooking, bool):
        raise TypeError("allow_overbooking must be a bool")
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef vector[cpp_PackedData] _ret
    cdef AllowOverbooking ab = (
        AllowOverbooking.YES if allow_overbooking else AllowOverbooking.NO
    )
    with nogil:
        _ret = cpp_unspill_partitions(
            move(_partitions),
            _br,
            ab,
        )
    return packed_data_vector_to_list(move(_ret), br)
