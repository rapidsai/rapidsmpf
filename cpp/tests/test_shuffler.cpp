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

#include <future>
#include <memory>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

#include "utils.hpp"

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {};

// test different `num_partitions` and `num_rows`.
INSTANTIATE_TEST_SUITE_P(
    Shuffler,
    NumOfPartitions,
    testing::Combine(
        testing::Range(1, 10),  // num_partitions
        testing::Range(1, 100, 9)  // num_rows
    )
);

TEST_P(NumOfPartitions, partition_and_pack) {
    int const num_partitions = std::get<0>(GetParam());
    int const num_rows = std::get<1>(GetParam());
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    auto chunks = rapidsmp::shuffler::partition_and_pack(
        expect, {1}, num_partitions, hash_fn, seed, stream, mr
    );

    // Convert to a vector
    std::vector<cudf::packed_columns> chunks_vector;
    for (auto& [_, chunk] : chunks) {
        chunks_vector.push_back(std::move(chunk));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result =
        rapidsmp::shuffler::unpack_and_concat(std::move(chunks_vector), stream, mr);

    // Compare the input table with the result. We ignore the row order by
    // sorting by their index (first column).
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST(MetadataMessage, round_trip) {
    auto metadata = iota_vector<uint8_t>(100);

    rapidsmp::shuffler::detail::Chunk expect(
        1, 2, true, 0, std::make_unique<std::vector<uint8_t>>(metadata), nullptr
    );

    // Extract the metadata from then chunk.
    auto msg = expect.to_metadata_message();
    EXPECT_TRUE(expect.metadata->empty());

    // Create a new chunk from the message.
    auto result = rapidsmp::shuffler::detail::Chunk::from_metadata_message(msg);

    // They should be identical.
    EXPECT_EQ(expect.pid, result.pid);
    EXPECT_EQ(expect.cid, result.cid);
    EXPECT_EQ(expect.expected_num_chunks, result.expected_num_chunks);
    EXPECT_EQ(expect.gpu_data, result.gpu_data);

    // The metadata should be identical to the original.
    EXPECT_EQ(metadata, *result.metadata);
}

using MemoryAvailableMap =
    std::unordered_map<rapidsmp::MemoryType, rapidsmp::BufferResource::MemoryAvailable>;

// Help function to get the `memory_available` argument for a `BufferResource`
// that prioritizes the specified memory type.
MemoryAvailableMap get_memory_available_map(rapidsmp::MemoryType priorities) {
    using namespace rapidsmp;

    // We set all memory types to use an available function that always return zero.
    BufferResource::MemoryAvailable always_zero = []() -> std::int64_t { return 0; };
    MemoryAvailableMap ret = {
        {MemoryType::DEVICE, always_zero}, {MemoryType::HOST, always_zero}
    };
    // And then set the prioritized memory type to use the max function.
    ret.at(priorities) = std::numeric_limits<std::int64_t>::max;
    return ret;
}

void test_shuffler(
    std::shared_ptr<rapidsmp::Communicator> const& comm,
    rapidsmp::shuffler::Shuffler& shuffler,
    rapidsmp::shuffler::PartID total_num_partitions,
    std::size_t total_num_rows,
    std::int64_t seed,
    cudf::hash_id hash_fn,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    // Every rank creates the full input table and all the expected partitions (also
    // partitions this rank might not get after the shuffle).
    cudf::table full_input_table = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expect_partitions, owner] = rapidsmp::shuffler::partition_and_split(
        full_input_table, {1}, total_num_partitions, hash_fn, seed, stream, mr
    );

    cudf::size_type row_offset = 0;
    cudf::size_type partiton_size = full_input_table.num_rows() / total_num_partitions;
    for (rapidsmp::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        // To simulate that `full_input_table` is distributed between multiple ranks,
        // we divided them into `total_num_partitions` number of partitions and pick
        // the partitions this rank should use as input. We pick using round robin but
        // any distribution would work (as long as no rows are picked by multiple ranks).
        // TODO: we should test different distributions of the input partitions.
        if (rapidsmp::shuffler::Shuffler::round_robin(comm, i) == comm->rank()) {
            cudf::size_type row_end = row_offset + partiton_size;
            if (i == total_num_partitions - 1) {
                // Include the reminder of rows in the very last partition.
                row_end = full_input_table.num_rows();
            }
            // Select the partition from the full input table.
            auto slice = cudf::slice(full_input_table, {row_offset, row_end}).at(0);
            // Hash the `slice` into chunks and pack (serialize) them.
            auto packed_chunks = rapidsmp::shuffler::partition_and_pack(
                slice, {1}, total_num_partitions, hash_fn, seed, stream, mr
            );
            // Add the chunks to the shuffle
            shuffler.insert(std::move(packed_chunks));
        }
        row_offset += partiton_size;
    }
    // Tell the shuffler that we have no more input partitions.
    for (rapidsmp::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        shuffler.insert_finished(i);
    }

    while (!shuffler.finished()) {
        auto finished_partition = shuffler.wait_any();
        auto packed_chunks = shuffler.extract(finished_partition);
        auto result =
            rapidsmp::shuffler::unpack_and_concat(std::move(packed_chunks), stream, mr);

        // We should only receive the partitions assigned to this rank.
        EXPECT_EQ(shuffler.partition_owner(comm, finished_partition), comm->rank());

        // Check the result while ignoring the row order.
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(result), sort_table(expect_partitions[finished_partition])
        );
    }
    shuffler.shutdown();
}

class MemoryAvailable_NumPartition
    : public cudf::test::BaseFixtureWithParam<std::tuple<MemoryAvailableMap, int, int>> {
};

// test different `memory_available` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    Shuffler,
    MemoryAvailable_NumPartition,
    testing::Combine(
        testing::ValuesIn(
            {get_memory_available_map(rapidsmp::MemoryType::HOST),
             get_memory_available_map(rapidsmp::MemoryType::DEVICE)}
        ),
        testing::Values(1, 2, 5, 10),  // total_num_partitions
        testing::Values(1, 9, 100)  // total_num_rows
    )
);

TEST_P(MemoryAvailable_NumPartition, round_trip) {
    MemoryAvailableMap const memory_available = std::get<0>(GetParam());
    rapidsmp::shuffler::PartID const total_num_partitions = std::get<1>(GetParam());
    std::size_t const total_num_rows = std::get<2>(GetParam());
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    rapidsmp::BufferResource br{mr, memory_available};

    MPI_Comm mpi_comm;
    RAPIDSMP_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(mpi_comm);
    rapidsmp::shuffler::Shuffler shuffler(
        comm,
        0,  // op_id
        total_num_partitions,
        stream,
        &br
    );

    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        comm,
        shuffler,
        total_num_partitions,
        total_num_rows,
        seed,
        hash_fn,
        stream,
        br.device_mr()
    ));

    RAPIDSMP_MPI(MPI_Comm_free(&mpi_comm));
}

// Test that the same communicator can be used concurrently by multiple shufflers in
// separate threads
class ConcurrentShuffleTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {
  public:
    void SetUp() override {
        num_shufflers = std::get<0>(GetParam());
        total_num_partitions = std::get<1>(GetParam());

        // these resources will be used by multiple threads to instantiate shufflers
        br = std::make_shared<rapidsmp::BufferResource>(mr());
        comm = std::make_shared<rapidsmp::MPI>(MPI_COMM_WORLD);
        stream = cudf::get_default_stream();
    }

    void TearDown() override {
        // make sure every process arrive at the end of the test case
        RAPIDSMP_MPI(MPI_Barrier(MPI_COMM_WORLD));
    }

    // test run for each thread. The test follows the same logic as
    // `MemoryAvailable_NumPartition` test, but without any memory limitations
    void RunTest(int t_id) {
        rapidsmp::shuffler::Shuffler shuffler(
            comm,
            t_id,  // op_id, use t_id as a proxy
            total_num_partitions,
            stream,
            br.get()
        );

        EXPECT_NO_FATAL_FAILURE(test_shuffler(
            comm,
            shuffler,
            total_num_partitions,
            100,  // total_num_rows
            t_id,  // seed
            cudf::hash_id::HASH_MURMUR3,
            stream,
            br->device_mr()
        ));
    }

    int num_shufflers;
    rapidsmp::shuffler::PartID total_num_partitions;

    std::shared_ptr<rapidsmp::BufferResource> br;
    std::shared_ptr<rapidsmp::Communicator> comm;
    rmm::cuda_stream_view stream;
};

// test different `num_shufflers` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    ConcurrentShuffle,
    ConcurrentShuffleTest,
    testing::Combine(
        testing::ValuesIn({1, 2, 4}),  // num_shufflers
        testing::ValuesIn({1, 10, 100})  // total_num_partitions
    ),
    [](const testing::TestParamInfo<ConcurrentShuffleTest::ParamType>& info) {
        return "num_shufflers_" + std::to_string(std::get<0>(info.param))
               + "__total_num_partitions_" + std::to_string(std::get<1>(info.param));
    }
);

TEST_P(ConcurrentShuffleTest, round_trip) {
    std::vector<std::future<void>> futures;
    futures.reserve(num_shufflers);

    for (int t_id = 0; t_id < num_shufflers; t_id++) {
        futures.push_back(std::async(std::launch::async, [this, t_id] {
            ASSERT_NO_FATAL_FAILURE(this->RunTest(t_id));
        }));
    }

    for (auto& f : futures) {
        ASSERT_NO_THROW(f.wait());
    }
}

TEST(Shuffler, SpillOnExtraction) {
    rapidsmp::shuffler::PartID const total_num_partitions = 2;
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();

    // Use a statistics memory resource.
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr(
        cudf::get_current_device_resource_ref()
    );

    // Create a buffer resource with an availabe device memory we can control
    // through the variable `device_memory_available`.
    std::int64_t device_memory_available{0};
    rapidsmp::BufferResource br{
        mr,
        {{rapidsmp::MemoryType::DEVICE,
          [&device_memory_available]() -> std::int64_t { return device_memory_available; }
        }}
    };
    EXPECT_EQ(
        br.memory_available(rapidsmp::MemoryType::DEVICE)(), device_memory_available
    );

    // Create a communicator of size 1, such that each shuffler will run locally.
    int rank;
    RAPIDSMP_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_Comm mpi_comm;
    RAPIDSMP_MPI(MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &mpi_comm));
    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(mpi_comm);
    EXPECT_EQ(comm->nranks(), 1);

    // Create a shuffler and input chunks.
    rapidsmp::shuffler::Shuffler shuffler(
        comm,
        0,  // op_id
        total_num_partitions,
        stream,
        &br
    );
    cudf::table input_table = random_table_with_index(seed, 1000, 0, 10);
    auto input_chunks = rapidsmp::shuffler::partition_and_pack(
        input_table, {1}, total_num_partitions, hash_fn, seed, stream, mr
    );

    // Insert spills does nothing when device memory is available, we start
    // with 2 device allocations.
    EXPECT_EQ(mr.get_allocations_counter().value, 2);
    shuffler.insert(std::move(input_chunks));
    // And we end with two 2 device allocations.
    EXPECT_EQ(mr.get_allocations_counter().value, 2);

    // Let's force spilling.
    device_memory_available = -1000;

    // But extract triggers spilling of the partition not being extracted.
    std::vector<cudf::packed_columns> output_chunks = shuffler.extract(0);
    EXPECT_EQ(mr.get_allocations_counter().value, 1);

    // Now insert also spills thus we end up with no device allocations.
    std::unordered_map<rapidsmp::shuffler::PartID, cudf::packed_columns> chunk;
    chunk.emplace(0, std::move(output_chunks.at(0)));
    shuffler.insert(std::move(chunk));
    EXPECT_EQ(mr.get_allocations_counter().value, 0);

    shuffler.shutdown();
    RAPIDSMP_MPI(MPI_Comm_free(&mpi_comm));
}
