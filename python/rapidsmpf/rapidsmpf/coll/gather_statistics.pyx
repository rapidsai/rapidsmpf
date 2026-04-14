# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport (Communicator, Rank,
                                                  cpp_Communicator)
from rapidsmpf.statistics cimport Statistics, cpp_Statistics


cdef extern from "<rapidsmpf/coll/gather_statistics.hpp>" nogil:
    vector[shared_ptr[cpp_Statistics]] cpp_gather_statistics \
        "rapidsmpf::coll::gather_statistics"(
            const shared_ptr[cpp_Communicator]& comm,
            int32_t op_id,
            const shared_ptr[cpp_Statistics]& stats,
            Rank root,
        ) except +ex_handler


def gather_statistics(
    Communicator comm not None,
    int32_t op_id,
    Statistics stats not None,
    Rank root = 0,
):
    """
    Gather statistics from all non-root ranks to the root rank.

    Non-root ranks serialize and send their statistics to the root rank.
    On root, the ``stats`` argument is ignored and the return value contains
    the deserialized statistics from every other rank. On non-root ranks the
    return value is an empty list.

    This is a blocking collective: all ranks must call this function.

    Parameters
    ----------
    comm
        The communicator.
    op_id
        Operation ID for tag disambiguation.
    stats
        The local statistics to send (ignored on root).
    root
        The root rank that collects the statistics (default 0).

    Returns
    -------
    On root: a list of deserialized Statistics from all non-root ranks.
    The gathered Statistics contain only stats, no memory records or formatters.
    On non-root ranks: an empty list.
    """
    cdef vector[shared_ptr[cpp_Statistics]] cpp_ret
    with nogil:
        cpp_ret = cpp_gather_statistics(
            comm._handle, op_id, stats._handle, root
        )

    cdef list ret = []
    cdef Statistics s
    for i in range(cpp_ret.size()):
        s = Statistics.__new__(Statistics)
        s._handle = cpp_ret[i]
        ret.append(s)
    return ret
