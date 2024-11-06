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

#include <mpi.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>

#include "utils.hpp"

Duration run(
    std::shared_ptr<rapidsmp::Communicator> comm,
    cudf::size_type const num_columns,
    cudf::size_type const num_local_rows,
    std::int32_t const min_val,
    std::int32_t const max_val,
    rapidsmp::shuffler::PartID const num_local_partitions,
    rapidsmp::shuffler::PartID const total_num_partitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    std::vector<cudf::table> input_partitions;
    for (rapidsmp::shuffler::PartID i = 0; i < num_local_partitions; ++i) {
        input_partitions.push_back(
            random_table(num_columns, num_local_rows, min_val, max_val, stream, mr)
        );
    }
    stream.synchronize();
    RAPIDSMP_MPI(MPI_Barrier(MPI_COMM_WORLD));

    auto const t0_elapsed = Clock::now();

    rapidsmp::shuffler::Shuffler shuffler(
        comm, total_num_partitions, rapidsmp::shuffler::Shuffler::round_robin, stream, mr
    );

    for (auto&& partition : input_partitions) {
        // Partition, pack, and insert this partition into the shuffler.
        shuffler.insert(rapidsmp::shuffler::partition_and_pack(
            partition,
            {0},
            total_num_partitions,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            mr
        ));
    }
    // Tell the shuffler that we have no more data.
    for (rapidsmp::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        shuffler.insert_finished(i);
    }

    std::vector<cudf::table> output_partitions;
    while (!shuffler.finished()) {
        auto finished_partition = shuffler.wait_any();
        auto packed_chunks = shuffler.extract(finished_partition);
        output_partitions.push_back(
            *rapidsmp::shuffler::unpack_and_concat(std::move(packed_chunks), stream, mr)
        );
    }
    stream.synchronize();
    auto const t1_elapsed = Clock::now();

    // Check the shuffle result
    for (const auto& output_partition : output_partitions) {
        auto [parts, owner] = rapidsmp::shuffler::partition_and_split(
            output_partition,
            {0},
            total_num_partitions,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            mr
        );
        RAPIDSMP_EXPECTS(
            std::count_if(
                parts.begin(),
                parts.end(),
                [](auto const& table) { return table.num_rows() > 0; }
            ) == 1,
            "all rows in an output partition should hash to the same"
        );
    }
    return t1_elapsed - t0_elapsed;
}

int main(int argc, char** argv) {
    rapidsmp::mpi::init(&argc, &argv);

    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(MPI_COMM_WORLD);
    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();

    // Benchmark arguments
    int const num_runs = 3;
    cudf::size_type const num_columns = 2;
    cudf::size_type const num_local_rows = 1 << 27;
    std::int32_t const min_val = 0;
    std::int32_t const max_val = num_local_rows;
    rapidsmp::shuffler::PartID const num_local_partitions = 2;
    rapidsmp::shuffler::PartID const total_num_partitions =
        num_local_partitions * comm->nranks();

    // Print benchmark/hardware info.
    {
        std::stringstream ss;
        std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
        CUDF_CUDA_TRY(cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), 0));
        cudaDeviceProp properties;
        CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Shuffle benchmark: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    PCI Bus ID: " << pci_bus_id << "\n";
        ss << "    Total Memory: " << to_mib(properties.totalGlobalMem) << " MiB\n";
        ss << "  Comm: " << *comm << "\n";
        log.warn(ss.str());
    }

    auto const local_nbytes =
        num_columns * num_local_rows * num_local_partitions * sizeof(std::int32_t);
    auto const total_nbytes = local_nbytes * comm->nranks();

    for (auto i = 0; i < num_runs; ++i) {
        auto elapsed =
            run(comm,
                num_columns,
                num_local_rows,
                min_val,
                max_val,
                num_local_partitions,
                total_num_partitions,
                stream,
                mr);
        log.warn(
            "elapsed: ",
            to_precision(elapsed),
            " sec, ",
            "local size: ",
            to_mib(local_nbytes),
            " MiB (",
            to_mib(local_nbytes / elapsed.count()),
            " MiB/s), ",
            "total size: ",
            to_mib(total_nbytes),
            " MiB (",
            to_mib(total_nbytes / elapsed.count()),
            " MiB/s), "
        );
    }

    RAPIDSMP_MPI(MPI_Finalize());
}
