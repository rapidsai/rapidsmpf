/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_stack.hpp"

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv) {
        RAPIDSMPF_EXPECTS(
            rapidsmpf::mpi::is_initialized() == true, "MPI is not initialized"
        );

        int rank, nranks;
        RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        RAPIDSMPF_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
        try {
            int option;
            while ((option = getopt(argc, argv, "C:r:w:c:n:p:o:m:l:gxh")) != -1) {
                switch (option) {
                case 'h':
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                           << "  -r <num>   Number of runs (default: 1)\n"
                           << "  -w <num>   Number of warmup runs (default: 0)\n"
                           << "  -c <num>   Number of columns in the input tables "
                              "(default: 1)\n"
                           << "  -n <num>   Number of rows per rank (default: 1M)\n"
                           << "  -p <num>   Number of partitions (input tables) per "
                              "rank (default: 1)\n"
                           << "  -o <num>   Number of output partitions per rank "
                              "(default: 1)\n"
                           << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                              "managed} "
                              "(default: cuda)\n"
                           << "  -l <num>   Device memory limit in MiB (default:-1, "
                              "disabled)\n"
                           << "  -g         Do hash partitining during data generation "
                              "(default: unset, hash partitioning during insertion)\n"
                           << "  -x         Enable memory profiler (default: disabled)\n"
                           << "  -h         Display this help message\n";
                        if (rank == 0) {
                            std::cerr << ss.str();
                        }
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                    }
                    break;
                case 'C':
                    comm_type = std::string{optarg};
                    if (!(comm_type == "mpi" || comm_type == "ucxx")) {
                        if (rank == 0) {
                            std::cerr << "-C (Communicator) must be one of {mpi, ucxx}"
                                      << std::endl;
                        }
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                    }
                    break;
                case 'r':
                    parse_integer(num_runs, optarg);
                    break;
                case 'w':
                    parse_integer(num_warmups, optarg);
                    break;
                case 'c':
                    parse_integer(num_columns, optarg);
                    break;
                case 'n':
                    parse_integer(num_local_rows, optarg);
                    break;
                case 'p':
                    parse_integer(num_local_partitions, optarg);
                    break;
                case 'o':
                    parse_integer(num_output_partitions, optarg);
                    break;
                case 'm':
                    rmm_mr = std::string{optarg};
                    if (!(rmm_mr == "cuda" || rmm_mr == "pool" || rmm_mr == "async"
                          || rmm_mr == "managed"))
                    {
                        if (rank == 0) {
                            std::cerr << "-m (RMM memory resource) must be one of "
                                         "{cuda, pool, async, managed}"
                                      << std::endl;
                        }
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                    }
                    break;
                case 'l':
                    parse_integer(device_mem_limit_mb, optarg);
                    break;
                case 'g':
                    hash_partition_with_datagen = true;
                    break;
                case 'x':
                    enable_memory_profiler = true;
                    break;
                case '?':
                    RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                    break;
                default:
                    RAPIDSMPF_FAIL("unknown option", std::invalid_argument);
                }
            }
            if (optind < argc) {
                RAPIDSMPF_FAIL("unknown option", std::invalid_argument);
            }
        } catch (std::exception const& e) {
            if (rank == 0) {
                std::cerr << "Error parsing arguments: " << e.what() << std::endl;
            }
            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
        }

        local_nbytes =
            num_columns * num_local_rows * num_local_partitions * sizeof(std::int32_t);
        total_nbytes = local_nbytes * static_cast<std::uint64_t>(nranks);
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

    void pprint(rapidsmpf::Communicator& comm) const {
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
        ss << "  -p " << num_local_partitions
           << " (number of input partitions per rank)\n";
        ss << "  -o " << num_output_partitions
           << " (number of output partitions per rank)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        if (device_mem_limit_mb >= 0) {
            ss << "  -l " << device_mem_limit_mb << " (device memory limit in MiB)\n";
        }
        if (enable_memory_profiler) {
            ss << "  -x (enable memory profiling)\n";
        }
        if (hash_partition_with_datagen) {
            ss << "  -g (hash partitioning during data generation)\n";
        }
        ss << "Local size: " << rapidsmpf::format_nbytes(local_nbytes) << "\n";
        ss << "Total size: " << rapidsmpf::format_nbytes(total_nbytes) << "\n";
        comm.logger().print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::uint32_t num_columns{1};
    std::uint64_t num_local_rows{1 << 20};
    rapidsmpf::shuffler::PartID num_local_partitions{1};
    rapidsmpf::shuffler::PartID num_output_partitions{1};
    std::string rmm_mr{"cuda"};
    std::string comm_type{"mpi"};
    std::uint64_t local_nbytes;
    std::uint64_t total_nbytes;
    bool enable_memory_profiler{false};
    bool hash_partition_with_datagen{false};
    std::int64_t device_mem_limit_mb{-1};
};

template <typename ShuffleInsertFn>
rapidsmpf::Duration do_run(
    rapidsmpf::shuffler::PartID const total_num_partitions,
    std::shared_ptr<rapidsmpf::Communicator>& comm,
    std::shared_ptr<rapidsmpf::ProgressThread>& progress_thread,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br,
    std::shared_ptr<rapidsmpf::Statistics>& statistics,
    ShuffleInsertFn&& shuffle_insert_fn
) {
    std::vector<cudf::table> output_partitions;
    output_partitions.reserve(total_num_partitions);

    auto const t0_elapsed = rapidsmpf::Clock::now();
    {
        RAPIDSMPF_NVTX_SCOPED_RANGE("Shuffling", total_num_partitions);
        rapidsmpf::shuffler::Shuffler shuffler(
            comm,
            progress_thread,
            0,  // op_id
            total_num_partitions,
            stream,
            br,
            statistics,
            rapidsmpf::shuffler::Shuffler::round_robin
        );

        // insert partitions into the shuffler
        shuffle_insert_fn(shuffler);

        // Tell the shuffler that we have no more data.
        for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
            shuffler.insert_finished(i);
        }

        while (!shuffler.finished()) {
            auto finished_partition = shuffler.wait_any();
            auto packed_chunks = shuffler.extract(finished_partition);
            output_partitions.push_back(*rapidsmpf::unpack_and_concat(
                std::move(packed_chunks), stream, br->device_mr()
            ));
        }
        stream.synchronize();
    }
    auto const t1_elapsed = rapidsmpf::Clock::now();

    // Check the shuffle result (this test only works for non-empty partitions
    // thus we only check large shuffles).
    if (args.num_local_rows >= 1000000) {
        for (const auto& output_partition : output_partitions) {
            auto [parts, owner] = rapidsmpf::partition_and_split(
                output_partition,
                {0},
                static_cast<std::int32_t>(total_num_partitions),
                cudf::hash_id::HASH_MURMUR3,
                cudf::DEFAULT_HASH_SEED,
                stream,
                br->device_mr()
            );
            RAPIDSMPF_EXPECTS(
                std::count_if(
                    parts.begin(),
                    parts.end(),
                    [](auto const& table) { return table.num_rows() > 0; }
                ) == 1,
                "all rows in an output partition should hash to the same"
            );
        }
    }
    return t1_elapsed - t0_elapsed;
}

std::vector<cudf::table> generate_input_partitions(
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    std::int32_t const min_val = 0;
    std::int32_t const max_val = args.num_local_rows;

    std::vector<cudf::table> input_partitions;
    input_partitions.reserve(args.num_local_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < args.num_local_partitions; ++i) {
        input_partitions.push_back(random_table(
            static_cast<cudf::size_type>(args.num_columns),
            static_cast<cudf::size_type>(args.num_local_rows),
            min_val,
            max_val,
            stream,
            mr
        ));
    }
    stream.synchronize();
    return input_partitions;
}

/**
 * @brief Runs shuffle by partitioning the input tables and inserting them into the
 * shuffler.
 *
 * This function generates random input tables and partitions them into the number of
 * input partitions specified by the user. It then inserts the partitions into the
 * shuffler and runs the shuffle. Each input partition will be partitioned into
 * `num_output_partitions * nranks`, resulting in, `num_local_partitions *
 * num_output_partitions * nranks` chunks being inserted into the shuffler. Each chunk
 * size will be `~num_local_rows/(num_output_partitions * nranks)` rows.
 *
 * @param comm Communicator for the shuffler
 * @param progress_thread Progress thread for the shuffler
 * @param args Command line arguments
 * @param stream CUDA stream for the shuffler
 * @param br Buffer resource for the shuffler
 * @param statistics Statistics for the shuffler
 * @return Duration of the run
 */
rapidsmpf::Duration run_hash_partition_inline(
    std::shared_ptr<rapidsmpf::Communicator>& comm,
    std::shared_ptr<rapidsmpf::ProgressThread>& progress_thread,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br,
    std::shared_ptr<rapidsmpf::Statistics>& statistics
) {
    rapidsmpf::shuffler::PartID const total_num_partitions =
        args.num_output_partitions
        * static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

    auto input_partitions = generate_input_partitions(args, stream, br->device_mr());

    RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));

    return do_run(
        total_num_partitions,
        comm,
        progress_thread,
        args,
        stream,
        br,
        statistics,
        [&](auto& shuffler) {
            for (auto&& partition : input_partitions) {
                // Partition, pack, and insert this partition into the shuffler.
                shuffler.insert(rapidsmpf::partition_and_pack(
                    partition,
                    {0},
                    static_cast<std::int32_t>(total_num_partitions),
                    cudf::hash_id::HASH_MURMUR3,
                    cudf::DEFAULT_HASH_SEED,
                    stream,
                    br->device_mr()
                ));
                partition.release();
            }
        }
    );
}

/**
 * @brief Runs shuffle by using already partitioned input tables.
 *
 * This is similar to the hash partitioning, but the input tables are already
 * partitioned before being inserted into the shuffler.
 *
 * @param comm Communicator for the shuffler
 * @param progress_thread Progress thread for the shuffler
 * @param args Command line arguments
 * @param stream CUDA stream for the shuffler
 * @param br Buffer resource for the shuffler
 * @param statistics Statistics for the shuffler
 * @return Duration of the run
 */

rapidsmpf::Duration run_hash_partition_with_datagen(
    std::shared_ptr<rapidsmpf::Communicator>& comm,
    std::shared_ptr<rapidsmpf::ProgressThread>& progress_thread,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br,
    std::shared_ptr<rapidsmpf::Statistics>& statistics
) {
    rapidsmpf::shuffler::PartID const total_num_partitions =
        args.num_output_partitions
        * static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

    auto input_partitions = generate_input_partitions(args, stream, br->device_mr());

    // generate input chunks
    std::vector<std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>>
        input_chunks;
    input_chunks.reserve(args.num_local_partitions);
    for (auto&& partition : input_partitions) {
        input_chunks.emplace_back(rapidsmpf::partition_and_pack(
            partition,
            {0},
            static_cast<std::int32_t>(total_num_partitions),
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            br->device_mr()
        ));
        partition.release();
    }

    input_partitions.clear();
    stream.synchronize();
    RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));

    return do_run(
        total_num_partitions,
        comm,
        progress_thread,
        args,
        stream,
        br,
        statistics,
        [&](auto& shuffler) {
            for (auto&& chunk : input_chunks) {
                shuffler.insert(std::move(chunk));
            }
        }
    );
}

int main(int argc, char** argv) {
    // Explicitly initialize MPI with thread support, as this is needed for both mpi and
    // ucxx communicators.
    int provided;
    RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    RAPIDSMPF_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );

    ArgumentParser args{argc, argv};

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    std::shared_ptr<rapidsmpf::Communicator> comm;
    if (args.comm_type == "mpi") {
        rapidsmpf::mpi::init(&argc, &argv);
        comm = std::make_shared<rapidsmpf::MPI>(MPI_COMM_WORLD, options);
    } else {  // ucxx
        comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options);
    }

    args.pprint(*comm);

    auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>(comm->logger());

    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);
    std::shared_ptr<rapidsmpf::RmmResourceAdaptor> stat_enabled_mr;
    if (args.enable_memory_profiler || args.device_mem_limit_mb >= 0) {
        stat_enabled_mr = set_device_mem_resource_with_stats();
    }

    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    if (args.device_mem_limit_mb >= 0) {
        memory_available[rapidsmpf::MemoryType::DEVICE] = rapidsmpf::LimitAvailableMemory{
            stat_enabled_mr.get(), args.device_mem_limit_mb << 20
        };
    }

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    rapidsmpf::BufferResource br{mr, std::move(memory_available)};

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
        ss << "    PCI Bus ID: " << pci_bus_id.substr(0, pci_bus_id.find('\0')) << "\n";
        ss << "    Total Memory: "
           << rapidsmpf::format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.print(ss.str());
    }

    // We start with disabled statistics.
    auto stats = std::make_shared<rapidsmpf::Statistics>(/* enable = */ false);

    std::vector<double> elapsed_vec;
    std::uint64_t const total_num_runs = args.num_warmups + args.num_runs;
    for (std::uint64_t i = 0; i < total_num_runs; ++i) {
        // Enable statistics for the last run.
        if (i == total_num_runs - 1) {
            stats = std::make_shared<rapidsmpf::Statistics>();
        }
        double elapsed;
        if (args.hash_partition_with_datagen) {
            elapsed = run_hash_partition_with_datagen(
                          comm, progress_thread, args, stream, &br, stats
            )
                          .count();
        } else {
            elapsed =
                run_hash_partition_inline(comm, progress_thread, args, stream, &br, stats)
                    .count();
        }
        std::stringstream ss;
        ss << "elapsed: " << rapidsmpf::to_precision(elapsed)
           << " sec | local throughput: "
           << rapidsmpf::format_nbytes(args.local_nbytes / elapsed)
           << "/s | global throughput: "
           << rapidsmpf::format_nbytes(args.total_nbytes / elapsed) << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log.print(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }

    RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));
    {
        auto const elapsed_mean = harmonic_mean(elapsed_vec);
        std::stringstream ss;
        ss << "means: " << rapidsmpf::to_precision(elapsed_mean)
           << " sec | local throughput: "
           << rapidsmpf::format_nbytes(args.local_nbytes / elapsed_mean)
           << "/s | global throughput: "
           << rapidsmpf::format_nbytes(args.total_nbytes / elapsed_mean) << "/s"
           << " | in_parts: " << args.num_local_partitions
           << " | out_parts: " << args.num_output_partitions
           << " | nranks: " << comm->nranks();
        if (args.enable_memory_profiler) {
            auto record = stat_enabled_mr->get_record();
            ss << " | device memory peak: " << rapidsmpf::format_nbytes(record.peak())
               << " | device memory total: "
               << rapidsmpf::format_nbytes(record.total() / total_num_runs) << " (avg)";
        }
        log.print(ss.str());
    }
    log.print(stats->report("Statistics (of the last run):"));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
