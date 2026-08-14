/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>

#include <ucxx/context.h>

#include <rapidsmpf/communicator/ucxx.hpp>

namespace rapidsmpf::ucxx::detail {

[[nodiscard]] std::shared_ptr<::ucxx::Context> create_context(ProgressMode progress_mode);

}  // namespace rapidsmpf::ucxx::detail
