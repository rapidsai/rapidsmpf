/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/coll/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf::coll;

extern Environment* GlobalEnvironment;

class BaseAllGatherTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stream = cudf::get_default_stream();
        br = std::make_unique<rapidsmpf::BufferResource>(rmm::mr::cuda_memory_resource{});
    }

    void TearDown() override {
        br = nullptr;
    }

    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
};

TEST_F(BaseAllGatherTest, timeout) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    EXPECT_THROW(
        std::ignore = allgather.wait_and_extract(
            AllGather::Ordered::NO, std::chrono::milliseconds{20}
        ),
        std::runtime_error
    );
    allgather.insert_finished();
    std::vector<rapidsmpf::PackedData> result;
    EXPECT_NO_THROW(
        result =
            allgather.wait_and_extract(AllGather::Ordered::NO, std::chrono::seconds{30})
    );
    EXPECT_EQ(result.size(), 0);
}

class AllGatherTest
    : public BaseAllGatherTest,
      public ::testing::WithParamInterface<std::tuple<int, int, AllGather::Ordered>> {
  protected:
    void SetUp() override {
        BaseAllGatherTest::SetUp();
        std::tie(n_elements, n_inserts, ordered) = GetParam();
    }

    int n_elements;
    int n_inserts;
    AllGather::Ordered ordered;
};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGather,
    AllGatherTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(0, 1, 10),  // n_inserts
        ::testing::Values(AllGather::Ordered::NO, AllGather::Ordered::YES)  // ordered
    ),
    [](const ::testing::TestParamInfo<AllGatherTest::ParamType>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "n_inserts"
               + std::to_string(std::get<1>(info.param)) + "_"
               + (std::get<2>(info.param) == AllGather::Ordered::YES ? "ordered"
                                                                     : "unordered");
    }
);

constexpr auto gen_offset(int i, int r) {
    return i * 10 + r;
};

TEST_P(AllGatherTest, basic_allgather) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto this_rank = comm->rank();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );
    if (n_inserts > 0) {
        EXPECT_EQ(n_inserts * comm->nranks(), results.size());

        if (ordered == AllGather::Ordered::YES) {
            // results vector should be ordered by rank and insertion order. Values should
            // look like:
            // rank0    |0... |10...|... * n_inserts
            // rank1    |1... |11...|... * n_inserts
            // ...
            // rank n-1 |(n-1)... |...   * n_inserts
            for (int r = 0; r < comm->nranks(); r++) {
                for (int i = 0; i < n_inserts; i++) {
                    auto& result = results[r * n_inserts + i];
                    int exp_offset = gen_offset(i, r);
                    EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                        std::move(result), n_elements, exp_offset, stream, *br
                    ));
                }
            }
        } else {  // unordered
            std::vector<int> exp_offsets;
            for (int i = 0; i < n_inserts * comm->nranks(); i++) {
                exp_offsets.emplace_back(gen_offset(i % n_inserts, i / n_inserts));
            }

            for (auto&& result : results) {
                if (n_elements == 0) {
                    EXPECT_EQ(result.metadata->size(), 0);
                    continue;
                }
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                auto it = std::ranges::find(exp_offsets, offset);
                EXPECT_NE(it, exp_offsets.end());
                exp_offsets.erase(it);
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, offset, stream, *br
                ));
            }
            if (n_elements != 0) {
                EXPECT_TRUE(exp_offsets.empty());
            }
        }
    } else {  // n_inserts == 0. No data is inserted.
        EXPECT_EQ(0, results.size());
    }
}

class AllGatherOrderedTest : public BaseAllGatherTest,
                             public ::testing::WithParamInterface<AllGather::Ordered> {};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGatherOrdered,
    AllGatherOrderedTest,
    ::testing::Values(AllGather::Ordered::NO, AllGather::Ordered::YES),  // ordered,
    [](auto const& info) {
        return info.param == AllGather::Ordered::YES ? "ordered" : "unordered";
    }
);

TEST_P(AllGatherOrderedTest, allgatherv) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    constexpr int n_inserts = 4;
    auto n_ranks = comm->nranks();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(this_rank, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );

    if (ordered == AllGather::Ordered::YES) {
        auto it = results.begin();
        for (int r = 0; r < n_ranks; r++) {
            for (int i = 0; i < n_inserts; i++) {
                auto& result = *it;
                EXPECT_EQ(r, static_cast<int>(result.metadata->size() / sizeof(int)));
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), r, gen_offset(i, r), stream, *br
                ));
                it++;
            }
        }
    } else {  // unordered
        for (auto&& result : results) {
            int n_elements = static_cast<int>(result.metadata->size() / sizeof(int));
            int offset =
                n_elements > 0 ? *reinterpret_cast<int*>(result.metadata->data()) : 0;
            EXPECT_NO_FATAL_FAILURE(
                validate_packed_data(std::move(result), n_elements, offset, stream, *br)
            );
        }
    }
}

TEST_P(AllGatherOrderedTest, non_uniform_inserts) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    auto n_inserts = this_rank;
    auto n_ranks = comm->nranks();

    constexpr int n_elements = 5;

    // call insert this_rank times
    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );

    // results should be a triangular number of elements
    EXPECT_EQ((n_ranks - 1) * n_ranks / 2, results.size());

    if (ordered == AllGather::Ordered::YES) {
        auto it = results.begin();
        for (int r = 0; r < n_ranks; r++) {
            for (int i = 0; i < r; i++) {
                auto& result = *it;
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, gen_offset(i, r), stream, *br
                ));
                it++;
            }
        }
    } else {  // unordered
        for (auto&& result : results) {
            if (result.data->size > 0) {
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, offset, stream, *br
                ));
            }
        }
    }
}

// Test that reusing an OpID after a completed allgather doesn't cause cross-matching of
// messages between the old and new collective.
//
// On rank 0 we inject a stream-ordered delay into device allocations so that received
// chunks stay "not ready" in the event loop's to_receive_ queue. The event loop keeps
// running (the host is not blocked). With small messages, other ranks can post via eager
// protocols, complete, and move on to the next allgather. Its control messages will then
// be matched on rank 0 by the blocked previous allgather, unless we correctly stop
// polling once we've seen all control messages.
TEST_F(BaseAllGatherTest, opid_reuse) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() == 1) {
        GTEST_SKIP() << "OpID reuse test requires multiple ranks";
    }

    constexpr int n_elements = 10;
    constexpr int n_inserts = 2;
    auto this_rank = comm->rank();

    // On rank 0, wrap the device MR with a delayed version.
    std::unique_ptr<rapidsmpf::BufferResource> delay_br;
    std::unique_ptr<AllGather> allgather;
    constexpr rapidsmpf::OpID op_id = 0;
    if (this_rank == 0) {
        // Recreate the buffer resource and allgather with the delayed MR.
        delay_br = std::make_unique<rapidsmpf::BufferResource>(
            DelayedMemoryResource{br->device_mr(), std::chrono::milliseconds(500)}
        );
        allgather =
            std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, delay_br.get());
    } else {
        allgather =
            std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, br.get());
    }

    for (int i = 0; i < n_inserts; i++) {
        allgather->insert(
            i, generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br)
        );
    }

    allgather->insert_finished();
    std::vector<rapidsmpf::PackedData> results1;
    EXPECT_NO_THROW(
        results1 =
            allgather->wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30})
    );
    // OK, it should be safe to reuse the opid now.
    allgather = std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, br.get());

    constexpr int second_offset = 1000;
    for (int i = 0; i < n_inserts; i++) {
        allgather->insert(
            i,
            generate_packed_data(
                n_elements, gen_offset(i + second_offset, this_rank), stream, *br
            )
        );
    }
    allgather->insert_finished();
    std::vector<rapidsmpf::PackedData> results2;
    EXPECT_NO_THROW(
        results2 =
            allgather->wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30})
    );
    ASSERT_EQ(static_cast<std::size_t>(n_inserts * comm->nranks()), results1.size());
    for (auto&& result : results1) {
        int offset = *reinterpret_cast<int*>(result.metadata->data());
        EXPECT_NO_FATAL_FAILURE(
            validate_packed_data(std::move(result), n_elements, offset, stream, *br)
        );
    }

    ASSERT_EQ(static_cast<std::size_t>(n_inserts * comm->nranks()), results2.size());

    // Every result must carry data from the second allgather.
    for (auto&& result : results2) {
        int offset = *reinterpret_cast<int*>(result.metadata->data());
        EXPECT_GE(offset, second_offset);
        EXPECT_NO_FATAL_FAILURE(
            validate_packed_data(std::move(result), n_elements, offset, stream, *br)
        );
    }
}

// Test that PostBox::spill() tracks the remaining spill need correctly across iterations,
// rather than passing the original amount to lower_bound every time.
//
// With chunks [20, 80, 90] and a request for 100 bytes: the first iteration spills 90
// (the largest chunk, since none covers 100 alone). The second must search for a chunk
// >= 10 and pick the 20-byte chunk, totalling 110.
TEST(PostBox, spill_uses_remaining_amount) {
    auto stream = cudf::get_default_stream();
    auto mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    auto br = std::make_unique<rapidsmpf::BufferResource>(*mr);

    rapidsmpf::coll::detail::PostBox postbox;

    auto make_chunk = [&](std::size_t size) {
        auto metadata =
            std::make_unique<std::vector<std::uint8_t>>(std::size_t{1}, std::uint8_t{0});
        auto res = br->reserve_or_fail(size, rapidsmpf::MemoryType::DEVICE);
        auto data = br->allocate(size, stream, res);
        return rapidsmpf::coll::detail::Chunk::from_packed_data(
            0,
            0,
            rapidsmpf::coll::detail::Chunk::INVALID_RANK,
            rapidsmpf::PackedData{std::move(metadata), std::move(data)}
        );
    };

    postbox.insert(make_chunk(20));
    postbox.insert(make_chunk(80));
    postbox.insert(make_chunk(90));

    EXPECT_EQ(postbox.spill(br.get(), 100), 110UL);
}
