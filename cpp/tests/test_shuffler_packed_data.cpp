/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <unordered_map>

#include <gtest/gtest.h>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include "environment.hpp"
#include "utils.hpp"

extern Environment* GlobalEnvironment;

TEST(ShufflerPackedData, basic_roundtrip) {
    auto const& comm = GlobalEnvironment->comm_;
    auto br = rapidsmpf::BufferResource::create(rmm::mr::cuda_memory_resource{});
    auto stream = br->stream_pool().get_stream();

    rapidsmpf::shuffler::Shuffler shuffler{
        comm, 0, 1, br.get(), rapidsmpf::shuffler::Shuffler::round_robin
    };

    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
    chunks.emplace(0, generate_packed_data(4, 0, stream, *br));
    shuffler.insert(std::move(chunks));
    shuffler.insert_finished();
    shuffler.wait(std::chrono::seconds{30});

    auto results = shuffler.extract(0);
    EXPECT_FALSE(results.empty());
}
