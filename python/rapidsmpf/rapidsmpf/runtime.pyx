# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for rapidsmpf::Runtime."""

from cython.operator cimport dereference as deref

from rapidsmpf.communicator.communicator cimport Logger
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics


cdef class Runtime:
    """
    Central runtime context owning configuration, statistics, and logging.

    ``Runtime`` is the single source of truth for :class:`.Options`,
    :class:`.Statistics`, and :class:`.Logger`. Create it once and pass it to
    every other RapidsMPF object (``BufferResource``, ``ProgressThread``,
    ``Context``, …) so that all components share the same configuration and
    statistics collector.

    Parameters
    ----------
    options
        Configuration options.  Environment variables are *not* inserted
        automatically; call ``options.insert_if_absent(get_environment_variables())``
        before passing if you want the usual env-var defaults.

    Examples
    --------
    >>> from rapidsmpf.config import Options, get_environment_variables
    >>> opts = Options()
    >>> opts.insert_if_absent(get_environment_variables())
    >>> rt = Runtime(opts)
    >>> rt.statistics.enabled
    False
    """
    def __init__(self, Options options not None):
        with nogil:
            self._handle = cpp_Runtime.from_options(options._handle)

    @classmethod
    def from_options(cls, Options options not None):
        """
        Construct a Runtime from configuration options.

        Parameters
        ----------
        options
            Configuration options.

        Returns
        -------
        Runtime
            A new Runtime instance.
        """
        return cls(options)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def reset(self, Options new_options not None):
        """
        Replace the runtime's configuration, statistics, and logger in-place.

        All three sub-objects are recreated from *new_options*.  Any previously
        obtained references to the old ``Statistics`` or ``Logger`` objects are
        invalidated after this call.

        .. warning::
            This method is **not** thread-safe.  Ensure no other threads are
            accessing this Runtime (or objects derived from it) while ``reset``
            is executing.

        Parameters
        ----------
        new_options
            New configuration options.
        """
        with nogil:
            deref(self._handle).reset(new_options._handle)

    @property
    def options(self):
        """
        The configuration options stored in this Runtime.

        Returns
        -------
        Options
            A snapshot of the current options (copy).
        """
        cdef Options ret = Options.__new__(Options)
        # options() returns a reference; copy it into a new Python Options.
        ret._handle = deref(self._handle).options()
        return ret

    @property
    def statistics(self):
        """
        The statistics collector owned by this Runtime.

        Returns
        -------
        Statistics
            The Statistics instance.  The returned object shares ownership with
            the Runtime; the Runtime must outlive the returned Statistics.
        """
        cdef Statistics ret = Statistics.__new__(Statistics)
        with nogil:
            ret._handle = deref(self._handle).statistics()
        return ret

    @property
    def logger(self):
        """
        The logger owned by this Runtime.

        Returns
        -------
        Logger
            The Logger instance.  The returned object shares ownership with the
            Runtime; the Runtime must outlive the returned Logger.
        """
        cdef Logger ret = Logger.__new__(Logger)
        with nogil:
            ret._handle = deref(self._handle).logger()
        return ret
