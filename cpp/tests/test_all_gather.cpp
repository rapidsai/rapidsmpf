/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/all_gather/all_gather.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf::all_gather;

extern Environment* GlobalEnvironment;

// Generate a packed data object with the given number of elements and offset.
// Both metadata and gpu_data contains the same data.
rapidsmpf::PackedData generate_packed_data(
    int n_elements,
    int offset,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br
) {
    auto values = iota_vector<int>(n_elements, offset);

    auto metadata = std::make_unique<std::vector<uint8_t>>(n_elements * sizeof(int));
    std::memcpy(metadata->data(), values.data(), n_elements * sizeof(int));

    auto data = std::make_unique<rmm::device_buffer>(
        values.data(), n_elements * sizeof(int), stream, br.device_mr()
    );

    return {std::move(metadata), br.move(std::move(data), stream)};
}

// Validate the packed data object by checking the metadata and gpu_data.
void validate_packed_data(
    rapidsmpf::PackedData const& packed_data,
    int n_elements,
    int offset,
    rmm::cuda_stream_view stream
) {
    auto const& metadata = *packed_data.metadata;
    EXPECT_EQ(metadata.size(), n_elements * sizeof(int));

    for (int i = 0; i < n_elements; i++) {
        int val;
        std::memcpy(&val, metadata.data() + i * sizeof(int), sizeof(int));
        EXPECT_EQ(offset + i, val);
    }

    EXPECT_EQ(packed_data.data->size, n_elements * sizeof(int));
    std::vector<uint8_t> copied_data(n_elements * sizeof(int));
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        copied_data.data(),
        packed_data.data->data(),
        n_elements * sizeof(int),
        cudaMemcpyDefault,
        stream
    ));

    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    EXPECT_EQ(metadata, copied_data);
}

class BaseAllGatherTest : public ::testing::Test {
  protected:
    void SetUp() override {
        GlobalEnvironment->barrier();

        stream = cudf::get_default_stream();
        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = std::make_unique<rapidsmpf::BufferResource>(mr.get());
        comm = GlobalEnvironment->comm_.get();

        all_gather = std::make_unique<AllGather>(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            0,
            stream,
            br.get()
        );
    }

    void TearDown() override {
        all_gather = nullptr;
        br = nullptr;
        mr = nullptr;
        GlobalEnvironment->barrier();
    }

    rmm::cuda_stream_view stream;
    rapidsmpf::Communicator* comm;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<AllGather> all_gather;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

// Test simple shutdown
TEST_F(BaseAllGatherTest, shutdown) {
    all_gather->shutdown();
}

class AllGatherTest : public BaseAllGatherTest,
                      public ::testing::WithParamInterface<std::tuple<int, int, bool>> {
  protected:
    void SetUp() override {
        BaseAllGatherTest::SetUp();
        std::tie(n_elements, n_inserts, ordered) = GetParam();
    }

    int n_elements;
    int n_inserts;
    bool ordered;
};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGather,
    AllGatherTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(0, 1, 10),  // n_inserts
        ::testing::Values(false, true)  // ordered
    ),
    [](const ::testing::TestParamInfo<AllGatherTest::ParamType>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "_n_inserts_"
               + std::to_string(std::get<1>(info.param)) + "_"
               + (std::get<2>(info.param) ? "ordered" : "unordered");
    }
);

constexpr auto gen_offset(int i, int r) {
    return i * 10 + r;
};

// Test basic all-gather
TEST_P(AllGatherTest, basic_all_gather) {
    auto this_rank = comm->rank();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        all_gather->insert(std::move(packed_data));
    }

    all_gather->insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    std::vector<uint64_t> n_chunks_per_rank;
    if (ordered) {
        std::tie(results, n_chunks_per_rank) = all_gather->wait_and_extract_ordered();
        EXPECT_EQ(comm->nranks(), n_chunks_per_rank.size());
    } else {
        results = all_gather->wait_and_extract();
    }

    EXPECT_TRUE(all_gather->finished());
    if (n_elements > 0 && n_inserts > 0) {  // only validate if there is data
        EXPECT_EQ(n_inserts * comm->nranks(), results.size());

        if (ordered) {
            EXPECT_TRUE(std::ranges::all_of(n_chunks_per_rank, [this](uint64_t n) {
                return n == static_cast<uint64_t>(n_inserts);
            }));

            // results vector should be ordered by rank and insertion order. Values should
            // look like:
            // rank0    |0... |10...|... * n_inserts
            // rank1    |1... |11...|... * n_inserts
            // ...
            // rank n-1 |(n-1)... |...   * n_inserts
            for (int r = 0; r < comm->nranks(); r++) {
                for (int i = 0; i < n_inserts; i++) {
                    auto const& result = results[r * n_inserts + i];
                    int exp_offset = gen_offset(i, r);
                    EXPECT_NO_FATAL_FAILURE(
                        validate_packed_data(result, n_elements, exp_offset, stream)
                    );
                }
            }
        } else {  // unordered
            std::vector<int> exp_offsets;
            for (int i = 0; i < n_inserts * comm->nranks(); i++) {
                exp_offsets.emplace_back(gen_offset(i % n_inserts, i / n_inserts));
            }

            for (auto const& result : results) {
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                auto it = std::ranges::find(exp_offsets, offset);
                EXPECT_NE(it, exp_offsets.end());
                exp_offsets.erase(it);
                EXPECT_NO_FATAL_FAILURE(
                    validate_packed_data(result, n_elements, offset, stream)
                );
            }
            EXPECT_TRUE(exp_offsets.empty());
        }
    } else {  // n_elements == 0 or n_inserts == 0. No data is inserted.
        EXPECT_EQ(0, results.size());
        if (ordered) {
            EXPECT_TRUE(std::ranges::all_of(n_chunks_per_rank, [](uint64_t n) {
                return n == 0;
            }));
        }
    }

    EXPECT_TRUE(all_gather->finished());
}

class AllGatherOrderedTest : public BaseAllGatherTest,
                             public ::testing::WithParamInterface<bool> {};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGatherOrdered,
    AllGatherOrderedTest,
    ::testing::Values(false, true),  // ordered,
    [](auto const& info) { return info.param ? "ordered" : "unordered"; }
);

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
        all_gather->insert(std::move(packed_data));
    }

    all_gather->insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    std::vector<uint64_t> n_chunks_per_rank;
    if (ordered) {
        std::tie(results, n_chunks_per_rank) = all_gather->wait_and_extract_ordered();
        EXPECT_EQ(n_ranks, n_chunks_per_rank.size());
        for (int r = 0; r < n_ranks; r++) {
            EXPECT_EQ(r, n_chunks_per_rank[r]);
        }
    } else {
        results = all_gather->wait_and_extract();
    }

    // results should be a triangular number of elements
    EXPECT_EQ((n_ranks - 1) * n_ranks / 2, results.size());

    if (ordered) {
        auto it = results.begin();
        for (int r = 0; r < n_ranks; r++) {
            for (int i = 0; i < r; i++) {
                auto const& result = *it;
                EXPECT_NO_FATAL_FAILURE(
                    validate_packed_data(result, n_elements, gen_offset(i, r), stream)
                );
                it++;
            }
        }
    } else {  // unordered
        for (auto const& result : results) {
            if (result.data->size > 0) {
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                EXPECT_NO_FATAL_FAILURE(
                    validate_packed_data(result, n_elements, offset, stream)
                );
            }
        }
    }

    EXPECT_TRUE(all_gather->finished());
}
