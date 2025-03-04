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
    """Base actor that initializes a shuffle operation upon setting up UCXX communication."""

    def __init__(self, nranks: int):
        """
        Initialize the actor.

        Parameters
        ----------
        nranks
            The number of ranks in the cluster.
        """
        super().__init__(nranks)
        self._device_mr: rmm.mr.DeviceMemoryResource | None = None
        self._buffer_resource: BufferResource | None = None

    def setup_worker(self, root_address_str: str) -> None:
        """
        Setup the UCXX communication and a shuffle operation.

        This method overrides the parent method. It will also call the create_device_mr
        and create_buffer_resource methods prior to creating the shuffler object.

        Parameters
        ----------
        root_address_str
            The address of the root node.
        """
        # First, call RapidsMPActor, which will set up the UCXX workers
        super().setup_worker(root_address_str)

        # create the device memory resource
        mr = self.create_device_mr()
        self.create_buffer_resource(mr)

    def create_device_mr(self) -> rmm.mr.DeviceMemoryResource:
        """
        Create a Device Memory Resource for this class.

        This will set `self.device_mr` property. This will be called by the
        `self.setup_worker` method. By default, it will create a CudaMemoryResource.

        Override this method to use a different device memory resourcef

        Returns
        -------
        rmm.mr.DeviceMemoryResource
            The device memory resource.
        """
        self._device_mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(self._device_mr)
        return self._device_mr

    def create_buffer_resource(
        self, device_mr: rmm.mr.DeviceMemoryResource
    ) -> BufferResource:
        """
        Create a Buffer Resource with this class' device memory resource.

        This will set `self.buffer_resource` property. This will be called by the
        `self.setup_worker` method. By default, it will create a BufferResource
        without any limits.

        Parameters
        ----------
        device_mr
            The device memory resource to use.

        Returns
        -------
        BufferResource
            The buffer resource.
        """
        self._buffer_resource = BufferResource(device_mr)
        return self._buffer_resource

    def initialize_shuffler(
        self,
        op_id: int = 0,
        total_num_partitions: int = -1,
    ) -> Shuffler:
        """
        Initialize a Shuffler using the communicator and buffer resource.

        Parameters
        ----------
        op_id
            The operation id. Default 0.
        total_num_partitions
            The total number of partitions. Default -1.

        Returns
        -------
        Shuffler
            New shuffler instance.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        if self.comm is None:
            raise RuntimeError("Communicator not initialized")

        return Shuffler(
            self.comm,
            op_id,
            total_num_partitions if total_num_partitions > 0 else self.nranks(),
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
