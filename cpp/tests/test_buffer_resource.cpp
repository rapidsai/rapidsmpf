/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>


using namespace rapidsmp;

constexpr std::size_t operator"" _KiB(unsigned long long n) {
    return n * (1 << 10);
}

TEST(BufferResource, ReservationOverbooking) {
    // Create a buffer resource that always have 10 KiB of available device memory.
    auto dev_mem_available = []() -> std::int64_t { return 10_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, dev_mem_available}}
    };
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
}

TEST(BufferResource, ReservationReleasing) {
    // Create a buffer resource that always have 10 KiB of available host and device
    // memory.
    auto dev_mem_available = []() -> std::int64_t { return 10_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(),
        {{MemoryType::DEVICE, dev_mem_available}, {MemoryType::HOST, dev_mem_available}}
    };
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Reserve all available host and device memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    auto [reserve2, overbooking2] = br.reserve(MemoryType::HOST, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserve2.size(), 10_KiB);
    EXPECT_EQ(overbooking2, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_THROW(br.release(reserve1, MemoryType::HOST, 10_KiB), std::invalid_argument);
    EXPECT_THROW(br.release(reserve2, MemoryType::DEVICE, 10_KiB), std::invalid_argument);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, MemoryType::DEVICE, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, MemoryType::DEVICE, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // A reservation is released when it goes out of scope.
    {
        auto [reserve, overbooking] = br.reserve(MemoryType::HOST, 10_KiB, true);
        EXPECT_EQ(reserve.size(), 10_KiB);
        EXPECT_EQ(overbooking, 10_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 20_KiB);
    }
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
}

TEST(BufferResource, LimitAvailableMemory) {
    rmm::mr::cuda_memory_resource mr_cuda;
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr{mr_cuda};
    auto stream = cudf::get_default_stream();

    // Create a buffer resource that uses `statistics_resource_adaptor` to limit
    // available device memory to 10 KiB.
    LimitAvailableMemory dev_mem_available{&mr, 10_KiB};
    BufferResource br{mr, {{MemoryType::DEVICE, dev_mem_available}}};
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available device memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allocating a Buffer also requires a reservation, which are then released.
    auto dev_buf1 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve1);
    EXPECT_EQ(dev_buf1->size, 10_KiB);
    EXPECT_EQ(reserve1.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    // Insufficent reservation for the allocation.
    EXPECT_THROW(
        br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve1), std::overflow_error
    );

    // Freeing a buffer increases the available but the reserved memory is unchanged.
    dev_buf1.reset();
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);

    // Moving buffers between memory types requires a reservation.
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf2 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve2);
    auto [reserve3, overbooking3] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    auto host_buf2 = br.move(MemoryType::HOST, std::move(dev_buf2), stream, reserve3);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // Moving buffers to the same memory type accepts an empty reservation.
    auto host_buf3 = br.move(MemoryType::HOST, std::move(host_buf2), stream, reserve3);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // But copying buffers always requires a reservation.
    EXPECT_THROW(
        br.copy(MemoryType::HOST, host_buf3, stream, reserve3), std::overflow_error
    );

    // The reservation must be of the correct memory type.
    auto [reserve4, overbooking4] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_THROW(
        br.copy(MemoryType::DEVICE, host_buf3, stream, reserve4), std::invalid_argument
    );
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // With the correct memory type, we can copy the buffer.
    auto [reserve5, overbooking5] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf3 = br.copy(MemoryType::DEVICE, host_buf3, stream, reserve5);
    EXPECT_EQ(dev_buf3->size, 10_KiB);
    EXPECT_EQ(reserve5.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);
}
