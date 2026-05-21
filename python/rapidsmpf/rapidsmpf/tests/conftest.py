# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.progress_thread import ProgressThread

if TYPE_CHECKING:
    from collections.abc import Generator

    from mpi4py import MPI

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for pytest."""
    parser.addoption(
        "--disable-mpi",
        action="store_true",
        default=False,
        help="Disable MPI tests",
    )


@pytest.fixture(scope="session")
def _mpi_disabled(pytestconfig: pytest.Config) -> bool:
    """Check if MPI tests are disabled via command line argument."""
    return bool(pytestconfig.getoption("--disable-mpi"))


def _get_mpi_module_or_skip(*, mpi_disabled: bool = False) -> MPI:
    """
    Return the `mpi4py.MPI` module or pytest.skip.

    Return the `mpi4py.MPI` module if MPI support is available, skip tests if
    `mpi4py.MPI` is unavailable or MPI tests are disabled via the --disable-mpi
    command line option.

    Parameters
    ----------
    mpi_disabled
        Whether MPI tests are disabled, by default False

    Returns
    -------
    The `mpi4py.MPI` module.

    Raises
    ------
    pytest.skip
        If MPI support is not available or --disable-mpi is specified
    """
    if "mpi" not in COMMUNICATORS:
        pytest.skip("No MPI support")

    if mpi_disabled:
        pytest.skip("MPI tests disabled")

    from mpi4py import MPI

    return MPI


@pytest.fixture(scope="session")
def _mpi_comm(*, _mpi_disabled: bool) -> Communicator:
    """
    Fixture for rapidsmpf's MPI communicator to use throughout the session.

    This fixture provides a session-wide `Communicator` instance that wraps
    the MPI world communicator (`MPI.COMM_WORLD`).

    Do not use this fixture directly, use the `comm` fixture instead.
    """
    MPI = _get_mpi_module_or_skip(mpi_disabled=_mpi_disabled)

    from rapidsmpf.communicator.mpi import new_communicator

    return new_communicator(
        MPI.COMM_WORLD, Options(get_environment_variables()), ProgressThread()
    )


@pytest.fixture(scope="session")
def _ucxx_comm() -> Communicator:
    """
    Fixture for rapidsmpf's UCXX communicator to use throughout the session.

    This fixture provides a session-wide `Communicator` instance that wraps
    the an underlying UCXX communicator.

    Do not use this fixture directly, use the `ucxx_comm` fixture instead.
    """
    from rapidsmpf.communicator.testing import ucxx_mpi_setup

    return ucxx_mpi_setup(None, Options(get_environment_variables()), ProgressThread())


@pytest.fixture(
    params=["mpi", "ucxx"],
)
def comm(request: pytest.FixtureRequest) -> Generator[Communicator]:
    """
    Fixture for a rapidsmpf communicator and setup, scoped for each test.
    """
    comm_name = request.param

    if "mpi" not in COMMUNICATORS:
        if comm_name == "mpi":
            pytest.skip("RapidsMPF not built with MPI support")
        if comm_name == "ucxx":
            pytest.skip(
                "RapidsMPF not built with MPI support, which is "
                "used to bootstrap this UCXX test"
            )
    if "ucxx" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with UCXX support")

    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    yield request.getfixturevalue(f"_{comm_name}_comm")
    MPI.COMM_WORLD.barrier()


@pytest.fixture
def device_mr() -> Generator[rmm.mr.CudaMemoryResource]:
    """
    Fixture for creating a new cuda memory resource and making it the
    current rmm resource temporarily.
    """
    prior_mr = rmm.mr.get_current_device_resource()
    try:
        mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        yield mr
    finally:
        rmm.mr.set_current_device_resource(prior_mr)


@pytest.fixture
def stream() -> Stream:
    """
    Fixture to get a CUDA stream.

    TODO: create a new stream compatible with the `device_mr` fixture. For now,
    we just return the default stream.
    """
    return DEFAULT_STREAM
