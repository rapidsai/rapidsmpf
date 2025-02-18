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

#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/communicator/ucx_utils.hpp>
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
    ArgumentParser(int argc, char* const* argv) {
        RAPIDSMP_EXPECTS(
            rapidsmp::mpi::is_initialized() == true, "MPI is not initialized"
        );

        int rank, nranks;
        RAPIDSMP_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        RAPIDSMP_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

        int option;
        while ((option = getopt(argc, argv, "hC:r:w:c:n:p:m:l:x")) != -1) {
            switch (option) {
            case 'h':
                {
                    std::stringstream ss;
                    ss << "Usage: " << argv[0] << " [options]\n"
                       << "Options:\n"
                       << "  -C <comm>  Communicator {mpi, ucx} (default: mpi)\n"
                       << "  -r <num>   Number of runs (default: 1)\n"
                       << "  -w <num>   Number of warmup runs (default: 0)\n"
                       << "  -c <num>   Number of columns in the input tables "
                          "(default: 1)\n"
                       << "  -n <num>   Number of rows per rank (default: 1M)\n"
                       << "  -p <num>   Number of partitions (input tables) per "
                          "rank (default: 1)\n"
                       << "  -m <mr>    RMM memory resource {cuda, pool, async} "
                          "(default: cuda)\n"
                       << "  -l <num>   Device memory limit in MiB (default:-1, "
                          "disabled) \n"
                       << "  -x         Enable memory profiler (default: disabled)\n"
                       << "  -h         Display this help message\n";
                    if (rank == 0) {
                        std::cerr << ss.str();
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                }
            case 'C':
                comm_type = std::string{optarg};
                if (!(comm_type == "mpi" || comm_type == "ucx")) {
                    if (rank == 0) {
                        std::cerr << "-C (Communicatpr) must be one of {mpi, ucx}"
                                  << std::endl;
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                }
                break;
            case 'r':
                num_runs = std::stoi(optarg);
                break;
            case 'w':
                num_warmups = std::stoi(optarg);
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
                    if (rank == 0) {
                        std::cerr << "-m (RMM memory resource) must be one of "
                                     "{cuda, pool, async}"
                                  << std::endl;
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                }
                break;
            case 'l':
                device_mem_limit_mb = std::stoll(optarg);
                break;
            case 'x':
                enable_memory_profiler = true;
                break;
            case '?':
                RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            default:
                throw std::runtime_error("Error parsing arguments.");
            }
        }
        if (optind < argc) {
            if (rank == 0) {
                std::cerr << "Unknown option: " << argv[optind] << std::endl;
            }
            RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }
        if (num_runs < 1) {
            if (rank == 0) {
                std::cerr << "-r (number of runs) must be greater than 0\n";
            }
            RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }
        if (num_local_rows < 1000) {
            if (rank == 0) {
                std::cerr << "-n (number of rows per rank) must be greater than 1000\n";
            }
            RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }
        local_nbytes =
            num_columns * num_local_rows * num_local_partitions * sizeof(std::int32_t);
        total_nbytes = local_nbytes * nranks;

        if (rmm_mr == "cuda") {
            if (rank == 0) {
                std::cout << "WARNING: using the default cuda memory resource "
                             "(-m cuda) might leak memory! A bug in UCX means "
                             "that device memory received through IPC is never "
                             "freed. Hopefully, this will be fixed in UCX v1.19."
                          << std::endl;
            }
        }
    }

    void pprint(rapidsmp::Communicator& comm) const {
        if (comm.rank() > 0) {
            return;
        }
        std::stringstream ss;
        ss << "Arguments:\n";
        ss << "  -c " << comm_type << " (communicator)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        ss << "  -c " << num_columns << " (number of columns)\n";
        ss << "  -n " << num_local_rows << " (number of rows per rank)\n";
        ss << "  -p " << num_local_partitions << " (number of partitions per rank)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        if (device_mem_limit_mb >= 0) {
            ss << "  -l " << device_mem_limit_mb << " (device memory limit in MiB)\n";
        }
        if (enable_memory_profiler) {
            ss << "  -x (enable memory profiling)\n";
        }
        ss << "Local size: " << rapidsmp::format_nbytes(local_nbytes) << "\n";
        ss << "Total size: " << rapidsmp::format_nbytes(total_nbytes) << "\n";
        comm.logger().info(ss.str());
    }

    int num_runs{1};
    int num_warmups{0};
    std::uint32_t num_columns{1};
    std::uint64_t num_local_rows{1 << 20};
    rapidsmp::shuffler::PartID num_local_partitions{1};
    std::string rmm_mr{"cuda"};
    std::string comm_type{"mpi"};
    std::uint64_t local_nbytes;
    std::uint64_t total_nbytes;
    bool enable_memory_profiler{false};
    std::int64_t device_mem_limit_mb{-1};
};

Duration run(
    std::shared_ptr<rapidsmp::Communicator> comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rapidsmp::BufferResource* br
) {
    std::int32_t const min_val = 0;
    std::int32_t const max_val = args.num_local_rows;
    rapidsmp::shuffler::PartID const total_num_partitions =
        args.num_local_partitions * comm->nranks();
    std::vector<cudf::table> input_partitions;
    for (rapidsmp::shuffler::PartID i = 0; i < args.num_local_partitions; ++i) {
        input_partitions.push_back(random_table(
            args.num_columns,
            args.num_local_rows,
            min_val,
            max_val,
            stream,
            br->device_mr()
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
            0,  // op_id
            total_num_partitions,
            stream,
            br,
            rapidsmp::shuffler::Shuffler::round_robin
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
                br->device_mr()
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
                std::move(packed_chunks), stream, br->device_mr()
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
            br->device_mr()
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
    // Explicitly initialize MPI with thread support, as this is needed for both mpi and
    // ucx communicators.
    int provided;
    RAPIDSMP_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    RAPIDSMP_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );

    ArgumentParser args{argc, argv};

    std::shared_ptr<rapidsmp::Communicator> comm;
    if (args.comm_type == "mpi") {
        rapidsmp::mpi::init(&argc, &argv);
        comm = std::make_shared<rapidsmp::MPI>(MPI_COMM_WORLD);
    } else {  // ucx
        comm = rapidsmp::ucxx::init_using_mpi(MPI_COMM_WORLD);
    }
    // barrier to synchronize all workers
    RAPIDSMP_MPI(MPI_Barrier(MPI_COMM_WORLD));

    args.pprint(*comm);

    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);
    std::shared_ptr<stats_dev_mem_resource> stat_enabled_mr;
    if (args.enable_memory_profiler || args.device_mem_limit_mb >= 0) {
        stat_enabled_mr = set_device_mem_resource_with_stats();
    }

    std::unordered_map<rapidsmp::MemoryType, rapidsmp::BufferResource::MemoryAvailable>
        memory_available{};
    if (args.device_mem_limit_mb >= 0) {
        memory_available[rapidsmp::MemoryType::DEVICE] = rapidsmp::LimitAvailableMemory{
            stat_enabled_mr.get(), args.device_mem_limit_mb << 20
        };
    }

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    rapidsmp::BufferResource br{mr, std::move(memory_available)};

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();

    // Print benchmark/hardware info.
    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
        CUDF_CUDA_TRY(cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev)
        );
        cudaDeviceProp properties;
        CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Hardware setup: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    Device number: " << cur_dev << "\n";
        ss << "    PCI Bus ID: " << pci_bus_id << "\n";
        ss << "    Total Memory: "
           << rapidsmp::format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.info(ss.str());
    }

    std::vector<double> elapsed_vec;
    for (auto i = 0; i < args.num_warmups + args.num_runs; ++i) {
        auto const elapsed = run(comm, args, stream, &br).count();
        std::stringstream ss;
        ss << "elapsed: " << rapidsmp::to_precision(elapsed)
           << " sec | local throughput: "
           << rapidsmp::format_nbytes(args.local_nbytes / elapsed)
           << "/s | total throughput: "
           << rapidsmp::format_nbytes(args.total_nbytes / elapsed) << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log.info(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }

    RAPIDSMP_MPI(MPI_Barrier(MPI_COMM_WORLD));
    {
        auto const elapsed_mean = harmonic_mean(elapsed_vec);
        std::stringstream ss;
        ss << "means: " << rapidsmp::to_precision(elapsed_mean)
           << " sec | local throughput: "
           << rapidsmp::format_nbytes(args.local_nbytes / elapsed_mean)
           << "/s | total throughput: "
           << rapidsmp::format_nbytes(args.total_nbytes / elapsed_mean) << "/s";
        if (args.enable_memory_profiler) {
            auto const counter = stat_enabled_mr->get_bytes_counter();
            ss << " | rmm device memory peak: " << rapidsmp::format_nbytes(counter.peak)
               << " | total: " << rapidsmp::format_nbytes(counter.total);
        }
        log.info(ss.str());
    }
    RAPIDSMP_MPI(MPI_Finalize());

    return 0;
}
