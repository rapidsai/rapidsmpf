# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

set(modules_only_rapidsmpf communicator.pyx)
set(modules_need_mpi mpi.pyx)
set(modules_need_ucxx ucxx.pyx)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${modules_only_rapidsmpf}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${modules_need_mpi}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf PRIVATE $<TARGET_NAME_IF_EXISTS:MPI::MPI_C>
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${modules_need_ucxx}"
  LINKED_LIBRARIES rapidsmpf::rapidsmpf PRIVATE ucxx::ucxx ucxx::python ucx::ucp
)
