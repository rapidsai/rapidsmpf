/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

#include "rapidsmp/buffer/buffer.hpp"
#include "utils.hpp"

using namespace rapidsmp;

TEST(BufferResource, LimitAvailableMemory) {
    rmm::mr::cuda_memory_resource mr_cuda;
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr{mr_cuda};

    LimitAvailableMemory mem_limit{&mr, 100};
    BufferResource br{mr, {{MemoryType::DEVICE, mem_limit}}};

    // Book all available memory
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 100, false);
    EXPECT_EQ(reserve1.size(), 100);
    EXPECT_EQ(overbooking1, 0);

    // Try to overbook
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 100, false);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 100);

    // Allow overbooking
    auto [reserve3, overbooking3] = br.reserve(MemoryType::DEVICE, 100, true);
    EXPECT_EQ(reserve3.size(), 100);
    EXPECT_EQ(overbooking3, 100);
}
