# Copyright (c) 2025, NVIDIA CORPORATION.

class Communicator:
    @property
    def rank(self) -> int: ...
    @property
    def nranks(self) -> int: ...
