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

constexpr std::size_t operator"" _KiB(unsigned long long n) {
    return n * (1 << 10);
}

TEST(BufferResource, LimitAvailableMemory) {
    rmm::mr::cuda_memory_resource mr_cuda;
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr{mr_cuda};
    auto stream = cudf::get_default_stream();

    LimitAvailableMemory dev_mem_available{&mr, 10_KiB};
    BufferResource br{mr, {{MemoryType::DEVICE, dev_mem_available}}};

    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Try to overbook.
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allow overbooking.
    auto [reserve3, overbooking3] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    EXPECT_EQ(reserve3.size(), 10_KiB);
    EXPECT_EQ(overbooking3, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // No host limit.
    auto [reserve4, overbooking4] = br.reserve(MemoryType::HOST, 10_KiB, false);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(overbooking4, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_THROW(br.release(reserve1, MemoryType::HOST, 10_KiB), std::invalid_argument);
    EXPECT_THROW(br.release(reserve4, MemoryType::DEVICE, 10_KiB), std::invalid_argument);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, MemoryType::DEVICE, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, MemoryType::DEVICE, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 15_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // We are still overbooking.
    auto [reserve5, overbooking5] = br.reserve(MemoryType::DEVICE, 5_KiB, true);
    EXPECT_EQ(reserve5.size(), 5_KiB);
    EXPECT_EQ(overbooking5, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Allocating a Buffer also requires a reservation, which are then released.
    auto dev_buf1 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve3);
    EXPECT_EQ(dev_buf1->size, 10_KiB);
    EXPECT_EQ(reserve3.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    // Insufficent reservation for the allocation.
    EXPECT_THROW(
        br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve3), std::overflow_error
    );

    // Freeing a buffer increases the available but the reserved memory is unchanged.
    dev_buf1.reset();
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // A reservation is released when it goes out of scope.
    {
        auto [reserve, overbooking] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
        EXPECT_EQ(reserve.size(), 10_KiB);
        EXPECT_EQ(overbooking, 10_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    }
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);


    // Moving buffers between memory types requires a reservation.
    auto [reserve6, overbooking6] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf2 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve6);
    auto [reserve7, overbooking7] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 20_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    auto host_buf2 = br.move(MemoryType::HOST, std::move(dev_buf2), stream, reserve7);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // Moving buffers to the same memory type accepts an empty reservation.
    auto host_buf3 = br.move(MemoryType::HOST, std::move(host_buf2), stream, reserve7);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // But copying buffers always requires a reservation.
    EXPECT_THROW(
        br.copy(MemoryType::HOST, host_buf3, stream, reserve7), std::overflow_error
    );

    // And the reservation must be of the correct memory type.
    auto [reserve8, overbooking8] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_THROW(
        br.copy(MemoryType::DEVICE, host_buf3, stream, reserve8), std::invalid_argument
    );
    EXPECT_EQ(reserve8.size(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 20_KiB);

    // With the correct memory type, we can copy the buffer.
    auto [reserve9, overbooking9] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf3 = br.copy(MemoryType::DEVICE, host_buf3, stream, reserve9);
    EXPECT_EQ(dev_buf3->size, 10_KiB);
    EXPECT_EQ(reserve9.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 20_KiB);
    EXPECT_EQ(dev_mem_available(), 0);
}
