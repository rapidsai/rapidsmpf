/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/owning_wrapper.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

#include "utils.hpp"


using namespace rapidsmpf;

TEST(SpillManager, SpillFunction) {
    // Create a buffer resource that report `mem_available` as the available memory.
    std::int64_t mem_available = 10_KiB;
    BufferResource br{
        cudf::get_current_device_resource_ref(),
        rapidsmpf::PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE,
          [&mem_available]() -> std::int64_t { return mem_available; }}}
    };
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 10_KiB);

    // Spill function that increases the available memory perfectly.
    SpillManager::SpillFunction func1 =
        [&mem_available](std::size_t amount) -> std::size_t {
        mem_available += amount;
        return amount;
    };
    br.spill_manager().add_spill_function(func1, /* priority = */ 1);
    EXPECT_EQ(br.spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 20_KiB);

    // Spill function that never spill any memory but has a higher priority.
    bool func2_called = false;
    SpillManager::SpillFunction func2 = [&func2_called](std::size_t) -> std::size_t {
        func2_called = true;
        return 0;
    };
    auto fid2 = br.spill_manager().add_spill_function(func2, /* priority = */ 2);
    EXPECT_EQ(br.spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_TRUE(func2_called);
    func2_called = false;
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 30_KiB);

    // Removing `func2` means it shouldn't run.
    br.spill_manager().remove_spill_function(fid2);
    EXPECT_EQ(br.spill_manager().spill(10_KiB), 10_KiB);
    EXPECT_FALSE(func2_called);
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 40_KiB);

    // If the headroom is already there, no spilling should be happening.
    EXPECT_EQ(br.spill_manager().spill_to_make_headroom(10_KiB), 0);
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 40_KiB);

    // If the headroom isn't there, we should spill to get the headroom.
    EXPECT_EQ(br.spill_manager().spill_to_make_headroom(100_KiB), 60_KiB);
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 100_KiB);

    // A negative headroom is allowed.
    EXPECT_EQ(br.spill_manager().spill_to_make_headroom(-100_KiB), 0);
    EXPECT_EQ(br.memory_available(MemoryType::DEVICE)(), 100_KiB);
}

TEST(SpillManager, PeriodicSpillCheck) {
    // Create a buffer resource that always trigger spilling (always reports
    // negative available memory).
    std::chrono::milliseconds period{1};
    BufferResource br{
        cudf::get_current_device_resource_ref(),
        PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE, []() -> std::int64_t { return -100_KiB; }}},
        period,
    };

    // Spill function that increases `mem` for each call.
    std::int64_t num_calls = 0;
    SpillManager::SpillFunction func =
        [&num_calls](std::size_t /* amount */) -> std::size_t { return ++num_calls; };
    br.spill_manager().add_spill_function(func, 0);

    std::this_thread::sleep_for(period * 100);
    // With no overhead, we should see 100 spill calls but we allow wiggle room.
    if (!is_running_under_valgrind()) {
        EXPECT_THAT(num_calls, testing::AllOf(testing::Gt(10), testing::Lt(200)));
    } else {
        // In valgrind, we cannot expect it to run more than once.
        EXPECT_THAT(num_calls, testing::AllOf(testing::Gt(1), testing::Lt(200)));
    }
}
