# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

@dataclasses.dataclass(frozen=True)
class ResourceBinding:
    rank: int | None
    gpu_id: int | None
    gpu_pci_bus_id: str
    cpu_affinity: str
    numa_nodes: list[int]
    ucx_net_devices: str

@dataclasses.dataclass(frozen=True)
class ExpectedBinding:
    cpu_affinity: str = ""
    memory_binding: list[int] = ...
    network_devices: list[str] = ...

@dataclasses.dataclass(frozen=True)
class BindingValidation:
    cpu_ok: bool
    numa_ok: bool
    ucx_ok: bool
    expected_ucx_devices: str
    def all_passed(self) -> bool: ...

def check_binding(gpu_id_hint: int | None = None) -> ResourceBinding: ...
def validate_binding(
    actual: ResourceBinding,
    expected: ExpectedBinding,
) -> BindingValidation: ...
def bind(
    gpu_id: int | None = None,
    *,
    cpu: bool = True,
    memory: bool = True,
    network: bool = True,
    verify: bool = True,
) -> None: ...
