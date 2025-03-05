# Copyright (c) 2025, NVIDIA CORPORATION.
"""Utils for the Ray integration."""

from __future__ import annotations

import rmm
import rmm.pylibrmm
import rmm.pylibrmm.stream

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.integrations.ray import RapidsMPActor
from rapidsmp.shuffler import Shuffler


class BaseShufflingActor(RapidsMPActor):
    """
    Base actor that initializes a shuffle operation upon setting up UCXX communication.

    Parameters
    ----------
    nranks
        The number of ranks in the cluster.
    """

    def __init__(self, nranks: int):
        super().__init__(nranks)
        self._device_mr: rmm.mr.DeviceMemoryResource | None = None
        self._buffer_resource: BufferResource | None = None

    def setup_worker(self, root_address_str: str) -> None:
        """
        Setup the UCXX communication and create device resources.

        This method overrides the parent method. It will create a Cuda memory resource
        and a buffer resource, which can be used to create Shuffler objects later.
        Override this method to use a different device memory resource.

        Parameters
        ----------
        root_address_str
            The address of the root node.
        """
        # First, call RapidsMPActor, which will set up the UCXX workers
        super().setup_worker(root_address_str)

        # create the device memory resource
        self._device_mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(self._device_mr)

        self._buffer_resource = BufferResource(self._device_mr)

    def create_shuffler(
        self,
        op_id: int,
        total_num_partitions: int | None = None,
    ) -> Shuffler:
        """
        Create a Shuffler using the communicator and buffer resource.

        Parameters
        ----------
        op_id
            The operation id.
        total_num_partitions
            The total number of partitions. By default, one partition per rank.

        Returns
        -------
        Shuffler
            New shuffler instance.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        if self.comm is None:
            raise RuntimeError("Communicator not initialized")

        if self.buffer_resource is None:
            raise RuntimeError("Buffer resource not initialized")

        return Shuffler(
            self.comm,
            op_id,
            total_num_partitions if total_num_partitions is not None else self.nranks(),
            DEFAULT_STREAM,
            self.buffer_resource,
        )

    @property
    def device_mr(self) -> rmm.mr.DeviceMemoryResource:
        """
        Get the device memory resource if initialized.

        Returns
        -------
        rmm.mr.DeviceMemoryResource
            The device memory resource.
        """
        if self._device_mr is None:
            raise RuntimeError("Device memory resource not initialized")

        return self._device_mr

    @property
    def buffer_resource(self) -> BufferResource:
        """
        Get the buffer resource if initialized.

        Returns
        -------
        BufferResource
            The buffer resource.
        """
        if self._buffer_resource is None:
            raise RuntimeError("Buffer resource not initialized")
        return self._buffer_resource
