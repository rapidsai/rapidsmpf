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


using namespace rapidsmp;

TEST(BufferResource, LimitAvailableMemory) {
    rmm::mr::cuda_memory_resource mr_cuda;
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr{mr_cuda};
    auto stream = cudf::get_default_stream();

    LimitAvailableMemory mem_limit{&mr, 100};
    BufferResource br{mr, {{MemoryType::DEVICE, mem_limit}}};

    EXPECT_EQ(mem_limit(), 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 100, false);
    EXPECT_EQ(reserve1.size(), 100);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Try to overbook.
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 100, false);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allow overbooking.
    auto [reserve3, overbooking3] = br.reserve(MemoryType::DEVICE, 100, true);
    EXPECT_EQ(reserve3.size(), 100);
    EXPECT_EQ(overbooking3, 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 200);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // No host limit.
    auto [reserve4, overbooking4] = br.reserve(MemoryType::HOST, 100, false);
    EXPECT_EQ(reserve4.size(), 100);
    EXPECT_EQ(overbooking4, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 200);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 100);

    // Cannot release the wrong memory type.
    EXPECT_THROW(br.release(reserve1, MemoryType::HOST, 100), std::invalid_argument);
    EXPECT_THROW(br.release(reserve4, MemoryType::DEVICE, 100), std::invalid_argument);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 200);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 100);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, MemoryType::DEVICE, 200), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 200);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 100);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, MemoryType::DEVICE, 50), 50);
    EXPECT_EQ(reserve1.size(), 50);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 150);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 100);

    // We are still overbooking.
    auto [reserve5, overbooking5] = br.reserve(MemoryType::DEVICE, 50, true);
    EXPECT_EQ(reserve5.size(), 50);
    EXPECT_EQ(overbooking5, 100);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 200);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 100);
}
