/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
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
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>
#include <rapidsmpf/utils.hpp>

#include "../utils/misc.hpp"
#include "../utils/rmm_stack.hpp"
#include "data_generator.hpp"

using namespace rapidsmpf;

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv) {
        RAPIDSMPF_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");

        int rank, nranks;
        RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        RAPIDSMPF_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
        try {
            int option;
            while ((option = getopt(argc, argv, "C:r:w:c:n:p:o:m:l:xh")) != -1) {
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

    void pprint(Communicator& comm) const {
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
        ss << "Local size: " << format_nbytes(local_nbytes) << "\n";
        ss << "Total size: " << format_nbytes(total_nbytes) << "\n";
        comm.logger().print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::uint32_t num_columns{1};
    std::uint64_t num_local_rows{1 << 20};
    shuffler::PartID num_local_partitions{1};
    shuffler::PartID num_output_partitions{1};
    std::string rmm_mr{"cuda"};
    std::string comm_type{"mpi"};
    std::uint64_t local_nbytes;
    std::uint64_t total_nbytes;
    bool enable_memory_profiler{false};
    std::int64_t device_mem_limit_mb{-1};
};

streaming::Node consumer(
    std::shared_ptr<streaming::Context> ctx,
    streaming::SharedChannel<streaming::TableChunk> ch_in
) {
    streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    while (true) {
        std::shared_ptr<streaming::TableChunk> table =
            co_await ch_in->receive_or(nullptr);
        if (table == nullptr) {
            break;
        }
    }
}

Duration run(
    std::shared_ptr<streaming::Context> ctx,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream
) {
    constexpr std::int32_t min_val = 0;
    constexpr std::int32_t max_val = 10;
    constexpr cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
    constexpr uint32_t seed = cudf::DEFAULT_HASH_SEED;
    rapidsmpf::shuffler::PartID const total_num_partitions =
        args.num_output_partitions
        * static_cast<rapidsmpf::shuffler::PartID>(ctx->comm()->nranks());
    constexpr OpID op_id = 0;

    // Create streaming pipeline.
    std::vector<streaming::Node> nodes;
    {
        auto ch1 = streaming::make_shared_channel<streaming::TableChunk>();
        nodes.push_back(
            streaming::node::random_table_generator(
                ctx,
                ch1,
                args.num_local_partitions,
                static_cast<cudf::size_type>(args.num_columns),
                static_cast<cudf::size_type>(args.num_local_rows),
                min_val,
                max_val,
                stream
            )
        );
        auto ch2 = streaming::make_shared_channel<streaming::PartitionMapChunk>();
        nodes.push_back(
            streaming::node::partition_and_pack(
                ctx,
                ch1,
                ch2,
                {0},
                static_cast<int>(total_num_partitions),
                hash_function,
                seed
            )
        );
        auto ch3 = streaming::make_shared_channel<streaming::PartitionVectorChunk>();
        nodes.push_back(
            streaming::node::shuffler(ctx, ch2, ch3, op_id, total_num_partitions)
        );
        auto ch4 = streaming::make_shared_channel<streaming::TableChunk>();
        nodes.push_back(streaming::node::unpack_and_concat(ctx, ch3, ch4));
        nodes.push_back(consumer(ctx, ch4));
    }
    auto const t0_elapsed = Clock::now();
    rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
    return Clock::now() - t0_elapsed;
}

int main(int argc, char** argv) {
    // Explicitly initialize MPI with thread support, as this is needed for both mpi
    // and ucxx communicators.
    int provided;
    RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    RAPIDSMPF_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );
    ArgumentParser args{argc, argv};

    // Initialize configuration options from environment variables.
    config::Options options{config::get_environment_variables()};

    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, options);
    } else {  // ucxx
        comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options);
    }

    args.pprint(*comm);

    RAPIDSMPF_EXPECTS(comm->nranks() == 1, "only single-rank runs are supported");

    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);
    std::shared_ptr<RmmResourceAdaptor> stat_enabled_mr;
    if (args.enable_memory_profiler || args.device_mem_limit_mb >= 0) {
        stat_enabled_mr = set_device_mem_resource_with_stats();
    }

    std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available{};
    if (args.device_mem_limit_mb >= 0) {
        memory_available[MemoryType::DEVICE] =
            LimitAvailableMemory{stat_enabled_mr.get(), args.device_mem_limit_mb << 20};
    }

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    BufferResource br(mr, std::move(memory_available));

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();

    // Print benchmark/hardware info.
    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
        RAPIDSMPF_CUDA_TRY(
            cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev)
        );
        cudaDeviceProp properties;
        RAPIDSMPF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Hardware setup: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    Device number: " << cur_dev << "\n";
        ss << "    PCI Bus ID: " << pci_bus_id.substr(0, pci_bus_id.find('\0')) << "\n";
        ss << "    Total Memory: " << format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.print(ss.str());
    }

    // We start with disabled statistics.
    auto stats = std::make_shared<Statistics>(/* enable = */ false);

    auto ctx = std::make_shared<streaming::Context>(options, comm, &br, stats);

    std::vector<double> elapsed_vec;
    std::uint64_t const total_num_runs = args.num_warmups + args.num_runs;
    for (std::uint64_t i = 0; i < total_num_runs; ++i) {
        // Enable statistics for the last run.
        if (i == total_num_runs - 1) {
            if (args.enable_memory_profiler) {
                stats = std::make_shared<Statistics>(stat_enabled_mr.get());
            } else {
                stats = std::make_shared<Statistics>(/* enable = */ true);
            }
        }
        double const elapsed = run(ctx, args, stream).count();
        std::stringstream ss;
        ss << "elapsed: " << to_precision(elapsed)
           << " sec | local throughput: " << format_nbytes(args.local_nbytes / elapsed)
           << "/s | global throughput: " << format_nbytes(args.total_nbytes / elapsed)
           << "/s";
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
        ss << "means: " << to_precision(elapsed_mean) << " sec | local throughput: "
           << format_nbytes(args.local_nbytes / elapsed_mean)
           << "/s | global throughput: "
           << format_nbytes(args.total_nbytes / elapsed_mean) << "/s"
           << " | in_parts: " << args.num_local_partitions
           << " | out_parts: " << args.num_output_partitions
           << " | nranks: " << comm->nranks();
        if (args.enable_memory_profiler) {
            auto record = stat_enabled_mr->get_main_record();
            ss << " | device memory peak: " << format_nbytes(record.peak())
               << " | device memory total: "
               << format_nbytes(
                      record.total() / static_cast<std::int64_t>(total_num_runs)
                  )
               << " (avg)";
        }
        log.print(ss.str());
    }
    log.print(stats->report("Statistics (of the last run):"));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
