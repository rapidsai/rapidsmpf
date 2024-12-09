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

#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/nvtx.hpp>
#include <rapidsmp/shuffler/partition.hpp>
#include <rapidsmp/shuffler/shuffler.hpp>
#include <rapidsmp/utils.hpp>

#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_stack.hpp"

class ArgumentParser {
  public:
    ArgumentParser(rapidsmp::Communicator& comm, int argc, char* const* argv) {
        int option;
        while ((option = getopt(argc, argv, "hr:c:n:p:m:")) != -1) {
            switch (option) {
            case 'h':
                {
                    std::stringstream ss;
                    ss << "Usage: " << argv[0] << " [options]\n"
                       << "Options:\n"
                       << "  -r <num>        Number of runs (default 1)\n"
                       << "  -c <num>        Number of columns in the input tables "
                          "(default 1)\n"
                       << "  -n <num>        Number of rows per rank (default 1M)\n"
                       << "  -p <num>        Number of partitions (input tables) per "
                          "rank (default 1)\n"
                       << "  -m <mr>         RMM memory resource {cuda, pool, async} "
                          "(cuda)\n"
                       << "  -h              Display this help message\n";
                    if (comm.rank() == 0) {
                        std::cerr << ss.str();
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                }
            case 'r':
                num_runs = std::stoi(optarg);
                break;
            case 'c':
                num_columns = std::stoul(optarg);
                break;
            case 'n':
                num_local_rows = std::stoull(optarg);
                break;
            case 'p':
                num_local_partitions = std::stoull(optarg);
                break;
            case 'm':
                rmm_mr = std::string{optarg};
                if (!(rmm_mr == "cuda" || rmm_mr == "pool" || rmm_mr == "async")) {
                    if (comm.rank() == 0) {
                        std::cerr << "-m (RMM memory resource) must be one of "
                                     "{cuda, pool, async}"
                                  << std::endl;
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                }
                if (rmm_mr == "cuda") {
                    if (comm.rank() == 0) {
                        std::cout << "WARNING: using the default cuda memory resource "
                                     "(-m cuda) might leak memory! A bug in UCX means "
                                     "that device memory received through IPC is never "
                                     "freed. Hopefully, this will be fixed in UCX v1.19."
                                  << std::endl;
                    }
                }
                break;
            case '?':
                RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            default:
                throw std::runtime_error("Error parsing arguments.");
            }
        }
        if (optind < argc) {
            if (comm.rank() == 0) {
                std::cerr << "Unknown option: " << argv[optind] << std::endl;
            }
            RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }
        if (num_local_rows < 1000) {
            if (comm.rank() == 0) {
                std::cerr << "-n (number of rows per rank) must be greater than 1000\n";
            }
            RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }
    }

    void pprint(rapidsmp::Communicator& comm) const {
        std::stringstream ss;
        ss << "Arguments:\n";
        ss << "  -r " << num_columns << " (number of runs)\n";
        ss << "  -c " << num_runs << " (number of columns)\n";
        ss << "  -n " << num_local_rows << " (number of rows per rank)\n";
        ss << "  -p " << num_local_partitions << " (number of partitions per rank)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        if (comm.rank() == 0) {
            std::cout << ss.str();
        }
    }

    int num_runs{1};
    std::uint32_t num_columns{1};
    std::uint64_t num_local_rows{1 << 20};
    rapidsmp::shuffler::PartID num_local_partitions{1};
    std::string rmm_mr{"cuda"};
};

Duration run(
    std::shared_ptr<rapidsmp::Communicator> comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    std::int32_t const min_val = 0;
    std::int32_t const max_val = args.num_local_rows;
    rapidsmp::shuffler::PartID const total_num_partitions =
        args.num_local_partitions * comm->nranks();
    std::vector<cudf::table> input_partitions;
    for (rapidsmp::shuffler::PartID i = 0; i < args.num_local_partitions; ++i) {
        input_partitions.push_back(random_table(
            args.num_columns, args.num_local_rows, min_val, max_val, stream, mr
        ));
    }
    stream.synchronize();
    RAPIDSMP_MPI(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<cudf::table> output_partitions;
    auto const t0_elapsed = Clock::now();
    {
        RAPIDSMP_NVTX_SCOPED_RANGE("Shuffling", total_num_partitions);
        rapidsmp::shuffler::Shuffler shuffler(
            comm,
            total_num_partitions,
            rapidsmp::shuffler::Shuffler::round_robin,
            stream,
            mr
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

        while (!shuffler.finished()) {
            auto finished_partition = shuffler.wait_any();
            auto packed_chunks = shuffler.extract(finished_partition);
            output_partitions.push_back(*rapidsmp::shuffler::unpack_and_concat(
                std::move(packed_chunks), stream, mr
            ));
        }
        stream.synchronize();
    }
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
    ArgumentParser args{*comm, argc, argv};
    args.pprint(*comm);

    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();

    // Print benchmark/hardware info.
    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
        CUDF_CUDA_TRY(cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev)
        );
        cudaDeviceProp properties;
        CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Shuffle benchmark: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    Device number: " << cur_dev << "\n";
        ss << "    PCI Bus ID: " << pci_bus_id << "\n";
        ss << "    Total Memory: "
           << rapidsmp::format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.warn(ss.str());
    }

    auto const local_nbytes = args.num_columns * args.num_local_rows
                              * args.num_local_partitions * sizeof(std::int32_t);
    auto const total_nbytes = local_nbytes * comm->nranks();

    for (auto i = 0; i < args.num_runs; ++i) {
        auto elapsed = run(comm, args, stream, mr);
        log.warn(
            "elapsed: ",
            rapidsmp::to_precision(elapsed.count()),
            " sec, ",
            "local size: ",
            rapidsmp::format_nbytes(local_nbytes),
            " (",
            rapidsmp::format_nbytes(local_nbytes / elapsed.count()),
            "/s), ",
            "total size: ",
            rapidsmp::format_nbytes(total_nbytes),
            " (",
            rapidsmp::format_nbytes(total_nbytes / elapsed.count()),
            "/s), "
        );
    }

    RAPIDSMP_MPI(MPI_Finalize());
}
