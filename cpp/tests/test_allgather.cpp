/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/allgather/allgather.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf::allgather;

extern Environment* GlobalEnvironment;

class BaseAllGatherTest : public ::testing::Test {
  protected:
    void SetUp() override {
        GlobalEnvironment->barrier();

        stream = cudf::get_default_stream();
        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = std::make_unique<rapidsmpf::BufferResource>(mr.get());
        comm = GlobalEnvironment->comm_.get();

        allgather = std::make_unique<AllGather>(
            GlobalEnvironment->comm_, GlobalEnvironment->progress_thread_, 0, br.get()
        );
    }

    void TearDown() override {
        allgather = nullptr;
        br = nullptr;
        mr = nullptr;
        GlobalEnvironment->barrier();
    }

    rmm::cuda_stream_view stream;
    rapidsmpf::Communicator* comm;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<AllGather> allgather;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

// Test simple shutdown
TEST_F(BaseAllGatherTest, shutdown) {}

TEST_F(BaseAllGatherTest, timeout) {
    EXPECT_THROW(
        std::ignore = allgather->wait_and_extract(
            AllGather::Ordered::NO, std::chrono::milliseconds{20}
        ),
        std::runtime_error
    );
    allgather->insert_finished();
    auto result = allgather->wait_and_extract();
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
    auto this_rank = comm->rank();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather->insert(i, std::move(packed_data));
    }

    allgather->insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    results = allgather->wait_and_extract(ordered);
    EXPECT_TRUE(allgather->finished());
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

    EXPECT_TRUE(allgather->finished());
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
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    constexpr int n_inserts = 4;
    auto n_ranks = comm->nranks();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(this_rank, gen_offset(i, this_rank), stream, *br);
        allgather->insert(i, std::move(packed_data));
    }

    allgather->insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    if (ordered == AllGather::Ordered::YES) {
        results = allgather->wait_and_extract(ordered);
    } else {
        do {
            std::ranges::move(allgather->extract_ready(), std::back_inserter(results));
        } while (!allgather->finished());
        std::ranges::move(allgather->extract_ready(), std::back_inserter(results));
    }
    EXPECT_EQ(n_ranks * n_inserts, results.size());

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

    EXPECT_TRUE(allgather->finished());
}

TEST_P(AllGatherOrderedTest, non_uniform_inserts) {
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    auto n_inserts = this_rank;
    auto n_ranks = comm->nranks();

    constexpr int n_elements = 5;

    // call insert this_rank times
    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather->insert(i, std::move(packed_data));
    }

    allgather->insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    results = allgather->wait_and_extract(ordered);

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

    EXPECT_TRUE(allgather->finished());
}
