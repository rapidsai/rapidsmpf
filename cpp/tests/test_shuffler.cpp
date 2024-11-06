/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <memory>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

#include "utils.hpp"

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<int> {};

// test the allowed bit widths for dictionary encoding
INSTANTIATE_TEST_SUITE_P(
    Shuffler, NumOfPartitions, testing::Range(1, 10), testing::PrintToStringParamName()
);

TEST_P(NumOfPartitions, partition_and_pack) {
    int const total_num_partitions = GetParam();
    std::int64_t const seed = 42;
    cudf::table expect = random_table_with_index(seed, 100, 0, 10);

    auto chunks =
        rapidsmp::shuffler::partition_and_pack(expect, {1}, total_num_partitions);

    // Convert to a vector
    std::vector<cudf::packed_columns> chunks_vector;
    for (auto& [_, chunk] : chunks) {
        chunks_vector.push_back(std::move(chunk));
    }

    auto result = rapidsmp::shuffler::unpack_and_concat(std::move(chunks_vector));

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

TEST_P(NumOfPartitions, round_trip) {
    std::int64_t const seed = 42;
    cudf::hash_id const hash_function = cudf::hash_id::HASH_MURMUR3;
    rapidsmp::shuffler::PartID const total_num_partitions = GetParam();

    MPI_Comm mpi_comm;
    RAPIDSMP_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(mpi_comm);
    rapidsmp::shuffler::Shuffler shuffler(comm, total_num_partitions);

    // Every rank creates the full input table and all the expected partitions (also
    // partitions this rank might not get after the shuffle).
    cudf::table full_input_table = random_table_with_index(seed, 100, 0, 10);
    auto [expect_partitions, owner] = rapidsmp::shuffler::partition_and_split(
        full_input_table, {1}, total_num_partitions, hash_function, seed
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
                slice, {1}, total_num_partitions, hash_function, seed
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
        auto result = rapidsmp::shuffler::unpack_and_concat(std::move(packed_chunks));

        // We should only receive the partitions assigned to this rank.
        EXPECT_EQ(shuffler.partition_owner(comm, finished_partition), comm->rank());

        // Check the result while ignoring the row order.
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(result), sort_table(expect_partitions[finished_partition])
        );
    }
    shuffler.shutdown();
    RAPIDSMP_MPI(MPI_Comm_free(&mpi_comm));
}
