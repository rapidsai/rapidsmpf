/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>

#include <gtest/gtest.h>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "environment.hpp"
#include "utils.hpp"

extern Environment* GlobalEnvironment;

TEST(AllGatherPackedData, basic_roundtrip) {
    auto const& comm = GlobalEnvironment->comm_;
    auto br = rapidsmpf::BufferResource::create(rmm::mr::cuda_memory_resource{});
    auto stream = br->stream_pool().get_stream();

    rapidsmpf::coll::AllGather allgather{comm, 0, br.get()};

    auto packed_data = generate_packed_data(4, 0, stream, *br);
    allgather.insert(0, std::move(packed_data));
    allgather.insert_finished();

    auto results = allgather.wait_and_extract(
        rapidsmpf::coll::AllGather::Ordered::NO, std::chrono::seconds{30}
    );
    EXPECT_FALSE(results.empty());
}
