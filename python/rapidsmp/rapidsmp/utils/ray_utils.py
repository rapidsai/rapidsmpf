# Copyright (c) 2025, NVIDIA CORPORATION.
"""Utils for the Ray integration."""

from __future__ import annotations

import rmm
import rmm.pylibrmm
import rmm.pylibrmm.stream
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.integrations.ray import RapidsMPActor
from rapidsmp.shuffler import Shuffler


class AbstractShufflingActor(RapidsMPActor):
    """Abstract actor that initializes a shuffle operation upon setting up UCXX communication."""

    def __init__(self, nranks: int, op_id: int = 0, total_nparts: int = -1):
        """Initialize the actor."""
        super().__init__(nranks)
        self._op_id: int = op_id
        self._total_nparts: int = total_nparts if total_nparts > 0 else nranks
        self._shuffler: Shuffler | None = None
        self._device_mr: rmm.mr.DeviceMemoryResource | None = None
        self._buffer_resource: BufferResource | None = None

    def setup_worker(self, root_address_str) -> None:
        """
        Setup the UCXX communication and a shuffle operation.

        This method overrides the parent method. It will also call the create_device_mr
        and create_buffer_resource methods prior to creating the shuffler object.
        """
        # First, call RapidsMPActor, which will set up the UCXX workers
        super().setup_worker(root_address_str)

        # create the device memory resource
        mr = self.create_device_mr()
        br = self.create_buffer_resource(mr)

        # initialize the shuffler
        self._shuffler = Shuffler(
            self.comm, self._op_id, self._total_nparts, DEFAULT_STREAM, br
        )

    def create_device_mr(self) -> rmm.mr.DeviceMemoryResource:
        """
        Create a Device Memory Resource for this class.

        This will set `self.device_mr` property. This will be called by the
        `self.setup_worker` method. By default, it will create a CudaMemoryResource.

        Override this method to use a different device memory resource
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
        """
        self._buffer_resource = BufferResource(device_mr)
        return self._buffer_resource

    @property
    def device_mr(self) -> rmm.mr.DeviceMemoryResource:
        """Returns the device memory resource if initialized."""
        if self._device_mr is None:
            raise RuntimeError("Device memory resource not initialized")

        return self._device_mr

    @property
    def buffer_resource(self) -> BufferResource:
        """Returns the buffer resource if initialized."""
        if self._buffer_resource is None:
            raise RuntimeError("Buffer resource not initialized")
        return self._buffer_resource

    @property
    def shuffler(self) -> Shuffler:
        """Returns the shuffler if initialized."""
        if self._shuffler is None:
            raise RuntimeError("Shuffler not initialized")

        return self._shuffler

    @property
    def total_nparts(self) -> int:
        """Returns the total number of partitions."""
        return self._total_nparts

    def shutdown(self) -> None:
        """Shuts down the shuffler."""
        if self._shuffler is not None:
            self.shuffler.shutdown()
            self._shuffler = None
