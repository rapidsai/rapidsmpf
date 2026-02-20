/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include "environment.hpp"
#include "utils.hpp"

extern Environment* GlobalEnvironment;

using namespace rapidsmpf;

/**
 * @brief Generate a random CUDA stream priority.
 *
 * @param random_generator A random number generator used to produce the priority.
 * @return A valid CUDA stream priority in the device range.
 */
int gen_stream_priority(std::mt19937 random_generator) {
    int least_priority = 0;  // numerically larger (often 0) => lower priority
    int greatest_priority = 0;  // numerically smaller (often negative) => higher priority
    RAPIDSMPF_CUDA_TRY(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority)
    );
    int num_priorities = least_priority - greatest_priority + 1;
    std::uniform_int_distribution<int> dist(0, num_priorities - 1);
    return greatest_priority + dist(random_generator);
}

// To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the
// shuffle shouldn't get near 30s.
constexpr std::chrono::seconds wait_timeout(30);

TEST(ShufflerManyStreams, Test) {
    std::mt19937 random_generator{42};
    constexpr std::size_t chunksize = 1 << 20;
    constexpr int num_partitions = 100;
    auto br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());

    // Create a CUDA stream for each partition.
    // To stress-test stream handling, assign random priorities so streams are more
    // likely to execute in mixed order.
    std::array<cudaStream_t, num_partitions> partition_streams{};
    for (shuffler::PartID pid = 0; pid < num_partitions; ++pid) {
        RAPIDSMPF_CUDA_TRY(cudaStreamCreateWithPriority(
            &partition_streams[pid],
            cudaStreamNonBlocking,
            gen_stream_priority(random_generator)
        ));
    }

    // Create the shuffler on `shuffler_stream`.
    rapidsmpf::shuffler::Shuffler shuffler(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        0,  // op_id
        num_partitions,
        br.get()
    );

    // Create each partition on its own CUDA stream.
    std::unordered_map<shuffler::PartID, PackedData> partitions;
    for (shuffler::PartID pid = 0; pid < num_partitions; ++pid) {
        partitions.insert(
            {pid, generate_packed_data(chunksize, pid, partition_streams[pid], *br)}
        );
    }

    // Insert all partitions.
    shuffler.insert(std::move(partitions));
    shuffler.insert_finished(iota_vector<rapidsmpf::shuffler::PartID>(num_partitions));

    // Extract and validate the partitions as they finishes.
    while (!shuffler.finished()) {
        auto pid = shuffler.wait_any(wait_timeout);
        std::vector<PackedData> partition_chunks = shuffler.extract(pid);
        for (PackedData& chunk : partition_chunks) {
            auto stream = chunk.data->stream();
            EXPECT_NO_FATAL_FAILURE(
                validate_packed_data(std::move(chunk), chunksize, pid, stream, *br)
            );
        }
    }

    // Cleanup streams
    for (auto& stream : partition_streams) {
        RAPIDSMPF_CUDA_TRY(cudaStreamDestroy(stream));
    }
}
