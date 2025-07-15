/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/all_gatherer/all_gatherer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf::experimental::all_gatherer;

extern Environment* GlobalEnvironment;

// Generate a packed data object with the given number of elements and offset.
// Both metadata and gpu_data contains the same data.
rapidsmpf::PackedData generate_packed_data(
    size_t n_elements,
    size_t offset,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    auto metadata = std::make_unique<std::vector<uint8_t>>();
    metadata->reserve(n_elements);
    for (size_t i = 0; i < n_elements; i++) {
        metadata->push_back(static_cast<uint8_t>(offset + i));  // wraps around 256
    }

    auto data = std::make_unique<rmm::device_buffer>(
        metadata->data(), metadata->size(), stream, mr
    );

    return {std::move(metadata), std::move(data)};
}

// Validate the packed data object by checking the metadata and gpu_data.
void validate_packed_data(
    rapidsmpf::PackedData const& packed_data,
    size_t n_elements,
    size_t offset,
    rmm::cuda_stream_view stream
) {
    auto const& metadata = *packed_data.metadata;
    EXPECT_EQ(metadata.size(), n_elements);
    for (size_t i = 0; i < n_elements; i++) {
        EXPECT_EQ(static_cast<uint8_t>(offset + i), metadata[i]);
    }

    std::vector<uint8_t> copied_data(n_elements);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        copied_data.data(),
        packed_data.gpu_data->data(),
        n_elements,
        cudaMemcpyDefault,
        stream
    ));

    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    EXPECT_EQ(metadata, copied_data);
}

class AllGathererTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<size_t, size_t>> {
  protected:
    void SetUp() override {
        std::tie(n_elements, n_inserts) = GetParam();
        stream = cudf::get_default_stream();
        br = std::make_unique<rapidsmpf::BufferResource>(mr());
        comm = GlobalEnvironment->comm_.get();

        all_gatherer = std::make_unique<AllGatherer>(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            0,
            stream,
            br.get()
        );
        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();

        all_gatherer.reset();
        br.reset();
    }

    size_t n_elements;
    size_t n_inserts;

    rmm::cuda_stream_view stream;
    rapidsmpf::Communicator* comm;
    std::unique_ptr<AllGatherer> all_gatherer;
    std::unique_ptr<rapidsmpf::BufferResource> br;
};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGatherer,
    AllGathererTest,
    ::testing::Combine(
        ::testing::Values(0, 10, 100),  // n_elements
        ::testing::Values(1, 10)  // n_inserts
    ),
    [](const ::testing::TestParamInfo<AllGathererTest::ParamType>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "_n_inserts_"
               + std::to_string(std::get<1>(info.param));
    }
);

// Test simple shutdown
TEST_P(AllGathererTest, shutdown) {
    all_gatherer->shutdown();
}

// Test basic all-gather
TEST_P(AllGathererTest, basic_all_gather) {
    for (size_t i = 0; i < n_inserts; i++) {
        auto packed_data = generate_packed_data(
            n_elements, static_cast<size_t>(comm->rank()), stream, mr()
        );
        all_gatherer->insert(std::move(packed_data));
    }

    all_gatherer->insert_finished();

    auto results = all_gatherer->wait_and_extract();

    EXPECT_TRUE(all_gatherer->finished());
    EXPECT_EQ(n_inserts * static_cast<size_t>(comm->nranks()), results.size());
    if (n_elements > 0) {  // only validate if there is data
        for (auto const& result : results) {
            size_t offset = result.metadata->at(0);
            EXPECT_NO_FATAL_FAILURE(
                validate_packed_data(result, n_elements, offset, stream)
            );
        }
    }
}
