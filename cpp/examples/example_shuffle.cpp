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

#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/partition.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/statistics.hpp>

#include "../benchmarks/utils/random_data.hpp"

// An example of how to use the shuffler.
int main(int argc, char** argv) {
    // In this example we use the MPI backed. For convenience, rapidsmp provides an
    // optional MPI-init function that initialize MPI with thread support.
    rapidsmp::mpi::init(&argc, &argv);

    // First, we have to create a Communicator, which we will use throughout the example.
    // Notice, if you want to do multiple shuffles concurrently, each shuffle should use
    // its own Communicator backed by its own MPI communicator.
    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(MPI_COMM_WORLD);

    // The Communicator provides a logger.
    auto& log = comm->logger();

    // We will use the same stream, memory, and buffer resource throughout the example.
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    rapidsmp::BufferResource br{mr};

    // Create a statistics instance for the shuffler that tracks useful information.
    auto stats = std::make_shared<rapidsmp::Statistics>(comm->nranks());

    // As input data, we use a helper function from the benchmark suite. It creates a
    // random cudf table with 2 columns and 100 rows. In this example, each MPI rank
    // creates its own local input and we only have one input per rank but each rank
    // could take any number of inputs.
    cudf::table local_input = random_table(2, 100, 0, 10, stream, mr);

    // The total number of inputs equals the number of ranks, in this case.
    rapidsmp::shuffler::PartID const total_num_partitions = comm->nranks();

    // We create a new shuffler instance, which represents a single shuffle. It takes
    // a Communicator, the total number of partitions, and a "owner function", which
    // map partitions to their destination ranks. All ranks must use the same owner
    // function, in this example we use the included round-robin owner function.
    rapidsmp::shuffler::Shuffler shuffler(
        comm,
        0,  // op_id
        total_num_partitions,
        stream,
        &br,
        stats,
        rapidsmp::shuffler::Shuffler::round_robin  // partition owner
    );

    // It is our own responsibility to partition and pack (serialize) the input for
    // the shuffle. The shuffler only handles raw host and device buffers. However, it
    // does provide a convenience function that hash partition a cudf table and packs
    // each partition. The result is a mapping of `PartID`, globally unique partition
    // identifiers, to their packed partitions.
    std::unordered_map<rapidsmp::shuffler::PartID, cudf::packed_columns> packed_inputs =
        rapidsmp::shuffler::partition_and_pack(
            local_input,
            {0},  // columns_to_hash
            total_num_partitions,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            mr
        );

    // Now, we can insert the packed partitions into the shuffler. This operation is
    // non-blocking and we can continue inserting new input partitions. E.g., a pipeline
    // could read, hash-partition, pack, and insert, one parquet-file at a time while the
    // distributed shuffle is being processed underneath.
    shuffler.insert(std::move(packed_inputs));

    // When we are finished inserting to a specific partition, we tell the shuffler.
    // Again, this is non-blocking and should be done as soon as we known that we don't
    // have more inputs for a specific partition. In this case, we are finished with all
    // partitions.
    for (rapidsmp::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        shuffler.insert_finished(i);
    }

    // Vector to hold the local results of the shuffle operation.
    std::vector<std::unique_ptr<cudf::table>> local_outputs;

    // Wait for and process the shuffle results for each partition.
    while (!shuffler.finished()) {
        // Block until a partition is ready and retrieve its partition ID.
        rapidsmp::shuffler::PartID finished_partition = shuffler.wait_any();

        // Extract the finished partition's data from the Shuffler.
        auto packed_chunks = shuffler.extract(finished_partition);

        // Unpack (deserialize) and concatenate the chunks into a single table using a
        // convenience function.
        local_outputs.push_back(
            rapidsmp::shuffler::unpack_and_concat(std::move(packed_chunks), stream, mr)
        );
    }
    // At this point, `local_outputs` contains the local result of the shuffle.
    // Let's log the result.
    log.info("Finished shuffle with ", local_outputs.size(), " local output partitions");

    // Shutdown the Shuffler explicitly or let it go out of scope for cleanup.
    shuffler.shutdown();

    // Finalize the execution, `RAPIDSMP_MPI` is a convenience macro that
    // checks for MPI errors.
    RAPIDSMP_MPI(MPI_Finalize());
}
