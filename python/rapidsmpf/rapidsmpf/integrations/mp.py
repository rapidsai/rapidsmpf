# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SPMD multiprocessing pool integration for RapidsMPF."""

from __future__ import annotations

import contextlib
import multiprocessing as _mp
import threading
from typing import TYPE_CHECKING, Any, Self

import cloudpickle
import ucxx._lib.libucxx as ucx_api

from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.core import WorkerContext, rmpf_worker_local_setup
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rrun.rrun import bind

if TYPE_CHECKING:
    from collections.abc import Callable


# Per-process WorkerContext — set inside each spawned worker by _worker_main.
_worker_context: WorkerContext | None = None


def get_worker_context() -> WorkerContext:
    """
    Retrieve the per-process :class:`~rapidsmpf.integrations.core.WorkerContext`.

    Must be called from within a worker process started by
    :class:`MultiprocessingPool`.  Analogous to
    :func:`rapidsmpf.integrations.single.get_worker_context`.

    Returns
    -------
    WorkerContext
        The worker context for the calling process.

    Raises
    ------
    RuntimeError
        If the worker context has not been initialised yet.

    See Also
    --------
    MultiprocessingPool
        Manages the pool of worker processes.
    """
    with WorkerContext.lock:
        if _worker_context is None:
            raise RuntimeError(
                "Worker context is not initialised. "
                "This function must be called from within a MultiprocessingPool worker."
            )
        return _worker_context


def _worker_main(
    rank: int,
    nranks: int,
    gpu_id: int,
    work_queue: Any,
    result_queue: Any,
    options_as_dict: dict[str, str],
    extra_sys_path: list[str],
    *,
    bind_cpu: bool,
    bind_memory: bool,
    bind_network: bool,
    bind_verify: bool,
    bind_raise_on_fail: bool,
) -> None:
    """
    Persistent worker-process event loop.

    This function is the ``target`` of each :class:`multiprocessing.Process`
    spawned by :class:`MultiprocessingPool`.  It must not be called directly.

    Parameters
    ----------
    rank
        Rank of this worker within the pool.
    nranks
        Total number of workers.
    gpu_id
        Physical GPU device index passed to :func:`rapidsmpf.rrun.rrun.bind`.
    work_queue
        Incoming task queue owned by the client.
    result_queue
        Outgoing result queue owned by the client.
    options_as_dict
        RapidsMPF options as a plain ``dict[str, str]``, reconstructed
        into :class:`~rapidsmpf.config.Options` on the worker side.
    extra_sys_path
        Copy of the parent's ``sys.path``, prepended to the worker's path
        so cloudpickle can resolve modules registered by pytest's
        ``--import-mode=importlib`` (e.g. ``python.rapidsmpf...test_mp``).
    bind_cpu
        Forward to :func:`~rapidsmpf.rrun.rrun.bind` ``cpu`` argument.
    bind_memory
        Forward to :func:`~rapidsmpf.rrun.rrun.bind` ``memory`` argument.
    bind_network
        Forward to :func:`~rapidsmpf.rrun.rrun.bind` ``network`` argument.
    bind_verify
        Forward to :func:`~rapidsmpf.rrun.rrun.bind` ``verify`` argument.
    bind_raise_on_fail
        When ``True``, a :class:`RuntimeError` from
        :func:`~rapidsmpf.rrun.rrun.bind` aborts the worker.
        When ``False`` (default), binding failures are silently ignored.
    """
    import sys

    global _worker_context  # noqa: PLW0603

    # Restore the parent's sys.path so cloudpickle can deserialise functions
    # whose modules were registered under paths added by pytest's importlib
    # mode (e.g. "python.rapidsmpf.rapidsmpf.tests.test_mp").
    for path in reversed(extra_sys_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    # ------------------------------------------------------------------ #
    # Phase 1: hardware binding                                            #
    # Must be the first action — before any CUDA / RMM initialisation.    #
    # ------------------------------------------------------------------ #
    try:
        bind(
            gpu_id=gpu_id,
            cpu=bind_cpu,
            memory=bind_memory,
            network=bind_network,
            verify=bind_verify,
        )
    except RuntimeError:
        if bind_raise_on_fail:
            result_queue.put(
                cloudpickle.dumps(
                    ("setup_error", RuntimeError(f"rrun.bind(gpu_id={gpu_id}) failed"))
                )
            )
            return

    # ------------------------------------------------------------------ #
    # Phase 2: create WorkerContext (initialises RMM / CUDA)              #
    # ------------------------------------------------------------------ #
    options = Options(options_as_dict)
    ctx = rmpf_worker_local_setup(rank, "mp_", options=options)

    # ------------------------------------------------------------------ #
    # Phase 3: bootstrap UCXX communicator                                #
    # ------------------------------------------------------------------ #
    try:
        if rank == 0:
            comm = new_communicator(
                nranks, None, None, options, ProgressThread(ctx.statistics)
            )
            root_addr_bytes: bytes = get_root_ucxx_address(comm)
            result_queue.put(cloudpickle.dumps(("root_addr", root_addr_bytes)))

        # All workers (including rank 0) wait for the root address relayed
        # back by the client.
        raw = work_queue.get()
        if raw is None:
            # Shutdown requested before setup completed.
            return
        msg_type, root_addr_bytes = cloudpickle.loads(raw)
        if msg_type != "setup":
            msg = f"Expected 'setup', got {msg_type!r}"
            raise RuntimeError(msg)  # noqa: TRY301

        if rank != 0:
            root_addr = ucx_api.UCXAddress.create_from_buffer(root_addr_bytes)
            comm = new_communicator(
                nranks, None, root_addr, options, ProgressThread(ctx.statistics)
            )

        # Synchronise all workers before signalling readiness.
        barrier(comm)

        ctx.comm = comm
        _worker_context = ctx
        result_queue.put(cloudpickle.dumps(("setup_done", rank)))

    except Exception as exc:
        result_queue.put(cloudpickle.dumps(("setup_error", exc)))
        return

    # ------------------------------------------------------------------ #
    # Phase 4: task-dispatch event loop                                   #
    # ------------------------------------------------------------------ #
    while True:
        try:
            raw = work_queue.get()
        except Exception:
            break

        if raw is None:
            # Poison-pill sentinel — clean shutdown.
            break

        try:
            func, args, kwargs = cloudpickle.loads(raw)
            result = func(*args, **kwargs)
            result_queue.put(cloudpickle.dumps(("ok", result)))
        except Exception as exc:
            result_queue.put(cloudpickle.dumps(("error", exc)))

    # ------------------------------------------------------------------ #
    # Phase 5: cleanup                                                    #
    # ------------------------------------------------------------------ #
    ctx.unregister_python_spill_callback()


class MultiprocessingPool:
    """
    SPMD execution pool backed by ``nranks`` persistent worker processes.

    Every call to :meth:`run` is **broadcast** to all workers — the same
    callable is executed on each process and results are returned in rank
    order.  This guarantees that every worker participates in every task,
    which is the correct semantics for SPMD workloads (shuffles,
    all-gathers, collective operations, etc.).

    The pool is the multiprocessing analogue of:

    - Dask's ``client.run(func)`` — fan-out to all workers
    - Ray's ``[actor._run.remote(func) for actor in rank_actors]``

    **Worker lifecycle**

    Each worker process (spawned with the ``spawn`` start method, which is
    required for CUDA safety):

    1. Calls :func:`rapidsmpf.rrun.rrun.bind` to pin itself to a GPU and
       its topologically-local CPUs, NUMA nodes, and NICs.
    2. Initialises a :class:`~rapidsmpf.integrations.core.WorkerContext`
       (which initialises RMM and allocates the buffer resource).
    3. Bootstraps a UCXX communicator connecting all ``nranks`` workers.
    4. Enters a persistent event loop, executing cloudpickle-serialised
       tasks received from the client via a dedicated per-worker
       :class:`~multiprocessing.Queue`.

    **Oversubscription**

    ``gpu_devices`` defaults to round-robin across available GPUs
    (``[rank % ngpus for rank in range(nranks)]``).  When ``nranks >
    ngpus``, multiple workers share the same physical GPU — each gets its
    own CUDA context.  The pool permits this by design, making it easy to
    test multi-rank logic on a single-GPU machine.

    Parameters
    ----------
    nranks
        Number of worker processes.
    options
        RapidsMPF configuration options.  Defaults to reading
        ``RAPIDSMPF_*`` environment variables.
    gpu_devices
        Physical GPU device IDs to assign to workers (one entry per
        rank).  Defaults to round-robin across available GPUs.  Pass a
        list of repeated IDs to deliberately oversubscribe (e.g.
        ``[0, 0, 0, 0]`` for four workers on one GPU).
    timeout
        Seconds to wait on any worker queue operation.  Applies to
        bootstrap as well as individual :meth:`run` calls.
    bind_cpu
        Pass ``cpu=bind_cpu`` to :func:`~rapidsmpf.rrun.rrun.bind`.
        Default ``True``.
    bind_memory
        Pass ``memory=bind_memory`` to :func:`~rapidsmpf.rrun.rrun.bind`.
        Default ``True``.
    bind_network
        Pass ``network=bind_network`` to
        :func:`~rapidsmpf.rrun.rrun.bind`.  Default ``True``.
    bind_verify
        Pass ``verify=bind_verify`` to :func:`~rapidsmpf.rrun.rrun.bind`.
        Default ``True``.  Set to ``False`` in environments where
        topology verification is unreliable (e.g. CI).
    bind_raise_on_fail
        When ``True``, a binding failure aborts the worker.
        When ``False`` (default), binding failures are silently ignored
        and the worker continues without hardware binding.

    Raises
    ------
    ValueError
        If ``gpu_devices`` length does not match ``nranks``.
    RuntimeError
        If any worker fails to start or bootstrap within ``timeout``
        seconds.

    Examples
    --------
    Context-manager style (recommended):

    >>> with MultiprocessingPool(nranks=2) as pool:  # doctest: +SKIP
    ...     ranks = pool.run(lambda: get_worker_context().comm.rank)

    Direct style:

    >>> pool = MultiprocessingPool(nranks=4)  # doctest: +SKIP
    >>> results = pool.run(some_module_level_function)  # doctest: +SKIP
    >>> pool.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        nranks: int,
        *,
        options: Options | None = None,
        gpu_devices: list[int] | None = None,
        timeout: float = 60.0,
        bind_cpu: bool = True,
        bind_memory: bool = True,
        bind_network: bool = True,
        bind_verify: bool = True,
        bind_raise_on_fail: bool = False,
    ) -> None:
        from cuda.core import system as cuda_system

        if options is None:
            options = Options(get_environment_variables())

        if gpu_devices is None:
            ngpus = cuda_system.get_num_devices()
            if ngpus == 0:
                raise RuntimeError("No CUDA devices found.")
            gpu_devices = [rank % ngpus for rank in range(nranks)]

        if len(gpu_devices) != nranks:
            raise ValueError(
                f"gpu_devices must have exactly nranks={nranks} entries, "
                f"got {len(gpu_devices)}."
            )

        self._nranks = nranks
        self._timeout = timeout
        self._lock = threading.Lock()

        mp_ctx = _mp.get_context("spawn")
        self._work_queues: list[Any] = [mp_ctx.Queue() for _ in range(nranks)]
        self._result_queues: list[Any] = [mp_ctx.Queue() for _ in range(nranks)]
        self._procs: list[_mp.Process] = []

        options_as_dict = options.get_strings()

        import sys
        from pathlib import Path

        # Build an extra_sys_path that workers can use to resolve the same
        # modules the parent can.  Two adjustments are needed:
        # 1. Replace '' (a CWD alias set only by `python -m`) with the
        #    absolute CWD so workers with a different CWD see the same path.
        # 2. Unconditionally prepend the absolute CWD so that module names
        #    like "python.rapidsmpf…tests.test_mp" — which pytest's
        #    --import-mode=importlib registers when running via the `pytest`
        #    binary (which does NOT add CWD to sys.path) — remain importable
        #    in spawned worker processes.
        cwd = str(Path.cwd())
        extra_sys_path = [cwd if p == "" else p for p in sys.path]
        if cwd not in extra_sys_path:
            extra_sys_path.insert(0, cwd)

        for rank in range(nranks):
            p = mp_ctx.Process(
                target=_worker_main,
                args=(
                    rank,
                    nranks,
                    gpu_devices[rank],
                    self._work_queues[rank],
                    self._result_queues[rank],
                    options_as_dict,
                    extra_sys_path,
                ),
                kwargs={
                    "bind_cpu": bind_cpu,
                    "bind_memory": bind_memory,
                    "bind_network": bind_network,
                    "bind_verify": bind_verify,
                    "bind_raise_on_fail": bind_raise_on_fail,
                },
                daemon=True,
                name=f"rapidsmpf-worker-{rank}",
            )
            p.start()
            self._procs.append(p)

        try:
            self._bootstrap(nranks, timeout)
        except Exception:
            self._kill_all()
            raise

    def _bootstrap(self, nranks: int, timeout: float) -> None:
        """Orchestrate UCXX communicator bootstrap across all workers."""
        # Phase 1: collect root UCXX address from rank 0.
        try:
            raw = self._result_queues[0].get(timeout=timeout)
        except Exception as exc:
            raise RuntimeError(
                "Timed out waiting for rank-0 worker to start."
            ) from exc

        msg_type, payload = cloudpickle.loads(raw)
        if msg_type == "setup_error":
            raise RuntimeError("Rank-0 worker failed during boot.") from payload
        if msg_type != "root_addr":
            raise RuntimeError(
                f"Unexpected message from rank-0 worker: {msg_type!r}"
            )

        root_addr_bytes: bytes = payload

        # Phase 2: relay the root address to all workers (including rank 0).
        setup_bytes = cloudpickle.dumps(("setup", root_addr_bytes))
        for q in self._work_queues:
            q.put(setup_bytes)

        # Phase 3: wait for every worker to complete setup.
        for i in range(nranks):
            try:
                raw = self._result_queues[i].get(timeout=timeout)
            except Exception as exc:
                raise RuntimeError(
                    f"Worker {i} timed out during UCXX bootstrap."
                ) from exc

            msg_type, payload = cloudpickle.loads(raw)
            if msg_type == "setup_error":
                raise RuntimeError(
                    f"Worker {i} failed during UCXX bootstrap."
                ) from payload
            if msg_type != "setup_done":
                raise RuntimeError(
                    f"Worker {i}: unexpected message {msg_type!r} during bootstrap."
                )

    def _kill_all(self) -> None:
        """Terminate all worker processes immediately."""
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        for p in self._procs:
            p.join(timeout=5)
        self._procs.clear()

    @property
    def nranks(self) -> int:
        """Number of worker processes in the pool."""
        return self._nranks

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> list[Any]:
        """
        Broadcast a callable to all workers and collect results.

        ``func`` is serialised with :mod:`cloudpickle` and dispatched to
        every worker's dedicated queue.  Each worker executes the callable
        independently and returns its result.  Results are collected in rank
        order (rank 0 first).

        This method is **thread-safe**: a per-instance :class:`threading.Lock`
        ensures that the broadcast → collect sequence is atomic with respect
        to other threads calling :meth:`run` or :meth:`shutdown`
        concurrently.  Only one call to :meth:`run` executes at a time.

        Parameters
        ----------
        func
            Callable to execute on every worker.  Supports lambdas,
            closures, and any other cloudpickle-serialisable callable.
        *args
            Positional arguments forwarded to ``func``.
        **kwargs
            Keyword arguments forwarded to ``func``.

        Returns
        -------
        list[Any]
            Results from each worker, ordered by rank.

        Raises
        ------
        RuntimeError
            If the pool has been shut down, or if any worker raises an
            exception during execution.
        """
        with self._lock:
            if not self._procs:
                raise RuntimeError("Pool has been shut down.")

            task_bytes = cloudpickle.dumps((func, args, kwargs))
            for q in self._work_queues:
                q.put(task_bytes)

            # Drain every result queue before raising so that failures in
            # some workers never leave dirty results in others, which would
            # poison the shared pool for subsequent run() calls.
            results: list[Any] = []
            errors: dict[int, Exception] = {}
            for i, rq in enumerate(self._result_queues):
                try:
                    raw = rq.get(timeout=self._timeout)
                except Exception as exc:
                    raise RuntimeError(
                        f"Worker {i} timed out waiting for a result."
                    ) from exc
                status, value = cloudpickle.loads(raw)
                if status == "error":
                    errors[i] = value
                else:
                    results.append(value)
            if errors:
                raise ExceptionGroup(
                    "One or more workers raised an exception during run().",
                    list(errors.values()),
                )
            return results

    def shutdown(self) -> None:
        """
        Shut down all worker processes gracefully.

        Sends a ``None`` sentinel to each worker's queue so it can
        clean up and exit, then joins all processes.  Idempotent: safe
        to call more than once.

        This method acquires the same lock as :meth:`run`, so it will
        block until any in-progress :meth:`run` call completes before
        beginning teardown.
        """
        with self._lock:
            if not self._procs:
                return

            for q in self._work_queues:
                with contextlib.suppress(Exception):
                    q.put(None)

            for p in self._procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()

            self._procs.clear()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()

    def __repr__(self) -> str:
        """Return a string representation of the pool."""
        status = "alive" if self._procs else "shut down"
        return f"MultiprocessingPool(nranks={self._nranks}, status={status!r})"
