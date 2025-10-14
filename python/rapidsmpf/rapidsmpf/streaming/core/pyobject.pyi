# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from rapidsmpf.streaming.core.channel import Message, Payload

class PyObjectPayload:
    @staticmethod
    def from_object(sequence_number: int, obj: Any) -> PyObjectPayload:
        """
        Create a PyObjectPayload from a Python object.

        Parameters
        ----------
        sequence_number
            Sequence number for this payload.
        obj
            Any Python object to wrap.

        Returns
        -------
        PyObjectPayload
            A new payload wrapping the given object.
        """

    @classmethod
    def from_message(cls, message: Message[Self]) -> Self:
        """
        Construct a PyObjectPayload by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PyObjectPayload.

        Returns
        -------
        A new PyObjectPayload extracted from the given message.
        """

    def into_message(self, message: Message[Self]) -> None:
        """Move this PyObjectPayload into a Message."""

    @property
    def sequence_number(self) -> int:
        """Return the sequence number of this payload."""

    def get_object(self) -> Any:
        """
        Get the wrapped Python object.

        Returns
        -------
        The Python object wrapped by this payload.
        """

if TYPE_CHECKING:
    # Check that PyObjectPayload implements Payload.
    _t1: PyObjectPayload
    _t2: Payload = _t1
