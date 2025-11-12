# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, Protocol, Self, TypeVar

from rapidsmpf.buffer.content_description import ContentDescription
from rapidsmpf.buffer.resource import MemoryReservation

PayloadT = TypeVar("PayloadT", bound="Payload")

class Payload(Protocol):
    """
    Protocol for the payload of a Message.

    Any object sent through a Channel must implement this protocol.
    It defines how to reconstruct the payload from a message and how to
    insert it back into a message.

    Methods
    -------
    from_message(message)
        Construct a payload instance by consuming a message.
    into_message(sequence_number, message)
        Insert the payload into a message. The payload instance is released
        in the process.
    """

    # Note: In the Cython implementations, these are all @staticmethod.
    # However, you cannot have a cdef classmethod, so we lie here.
    # We can't annotate as a staticmethod because that would give us a
    # function type:
    #
    #     forall T. from_message :: Message[T] -> T
    #
    # i.e. every implementation must be able to unwrap an arbitrary
    # Message[T] and produce a T.
    #
    # But each implementation type I can only deliver a function type:
    #
    #    from_message :: Message[I] -> I
    #
    # for some concrete I, and we've lost the universal quantifier, so we
    # can't pass a Message[Foo] to an implementation I, even though the
    # staticmethod protocol says that we should be able to.
    #
    # With a classmethod, the function type is now:
    #
    #    forall T. from_message :: type[T] -> Message[T] -> T
    #
    # i.e. every implementation must be able to unwrap a Message[T] and
    # produce a T iff type[T] is the class type.
    #
    # The concrete implementation is then:
    #
    #    from_message :: type[I] -> Message[I] -> I
    #
    # and this satisfies the protocol because instead of losing the
    # universal quantifier we've specialised onto the case where T = I.
    @classmethod
    def from_message(cls: type[PayloadT], message: Message[PayloadT]) -> PayloadT: ...
    def into_message(
        self: PayloadT, sequence_number: int, message: Message[PayloadT]
    ) -> None: ...

class Message(Generic[PayloadT]):
    def __init__(self, sequence_number: int, payload: PayloadT): ...
    def empty(self) -> bool: ...
    @property
    def sequence_number(self) -> int: ...
    def get_content_description(self) -> ContentDescription: ...
    def copy_cost(self) -> int: ...
    def copy(self, reservation: MemoryReservation) -> Self: ...
