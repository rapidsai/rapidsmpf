# Copyright (c) 2025, NVIDIA CORPORATION.
"""Utils for the Ray integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import rmm
import rmm.pylibrmm
import rmm.pylibrmm.stream

from rapidsmp.buffer.resource import BufferResource
from rapidsmp.integrations.ray import RapidsMPActor
from rapidsmp.shuffler import Shuffler

if TYPE_CHECKING:
    from rapidsmp.statistics import Statistics


class BaseShufflingActor(RapidsMPActor):
    """
    Base actor that initializes a shuffle operation upon setting up UCXX communication.

    Parameters
    ----------
    nranks
        Number of ranks.
    """

    def __init__(self, nranks: int):
        super().__init__(nranks)
        self._default_br: BufferResource | None = None

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """
        Setup the UCXX communication and initializes the default buffer resource with the current RMM device resource.

        This method overrides the parent method. It will create a Cuda memory resource
        and a buffer resource, which can be used to create Shuffler objects later.
        Override this method to use a different device memory resource.

        Parameters
        ----------
        root_address_bytes
            The address of the root node.
        """
        # First, call RapidsMPActor, which will set up the UCXX workers
        super().setup_worker(root_address_bytes)

        # Initialize the default buffer resource using the current rmm device resource.
        self._default_br = BufferResource(rmm.mr.get_current_device_resource())

    def create_shuffler(
        self,
        op_id: int,
        total_num_partitions: int | None = None,
        stream: rmm.pylibrmm.stream.Stream | None = None,
        buffer_resource: BufferResource | None = None,
        statistics: Statistics | None = None,
    ) -> Shuffler:
        """
        Create a Shuffler using the communicator and buffer resource.

        Parameters
        ----------
        op_id
            The operation id which is used to identify the shuffle operation. If there are multiple
            concurrent shuffle operations, each should be uniquely identified. Op ID may be reused, after shutting down a shuffler.
        total_num_partitions
            The total number of partitions. By default, one partition per rank.
        stream
            Stream to use for the shuffle operation. If None, the default stream will be used.
        buffer_resource
            The buffer resource to use for the shuffle operation. If None, the default buffer resource will be used.
        statistics
            Statistics object to use.

        Returns
        -------
        Shuffler
            New shuffler instance.
        """
        if self.comm is None:
            raise RuntimeError("Communicator not initialized")

        if stream is None:
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            stream = DEFAULT_STREAM

        return Shuffler(
            self.comm,
            op_id,
            total_num_partitions if total_num_partitions is not None else self.nranks(),
            stream,
            buffer_resource
            if buffer_resource is not None
            else self.default_buffer_resource,
            statistics=statistics,
        )

    @property
    def default_buffer_resource(self) -> BufferResource:
        """
        Get the default buffer resource if initialized.

        Returns
        -------
        BufferResource
            The buffer resource.
        """
        if self._default_br is None:
            raise RuntimeError("Buffer resource not initialized")
        return self._default_br
