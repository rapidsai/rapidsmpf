/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <iostream>

#include <mpi.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_stack.hpp"


using namespace rapidsmpf;

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv, bool use_mpi = true) {
        int rank = 0;
        int nranks = 1;

        if (use_mpi) {
            RAPIDSMPF_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");
            RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
            RAPIDSMPF_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
        } else {
            // When not using MPI, expect to be using bootstrap mode (rrun)
            nranks = rapidsmpf::bootstrap::get_nranks();
        }

        try {
            int option;
            while ((option = getopt(argc, argv, "hC:O:r:w:n:p:m:M:")) != -1) {
                switch (option) {
                case 'h':
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                           << "             ucxx automatically detects launcher (mpirun "
                              "or rrun)\n"
                           << "  -O <op>    Operation {all-to-all} (default: "
                              "all-to-all)\n"
                           << "  -n <num>   Message size in bytes (default: 1M)\n"
                           << "  -p <num>   Number of concurrent operations, e.g. number"
                              " of  concurrent all-to-all operations (default: 1)\n"
                           << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                              "managed} "
                              "(default: pool)\n"
                           << "  -r <num>   Number of runs (default: 1)\n"
                           << "  -w <num>   Number of warmup runs (default: 0)\n"
#ifdef RAPIDSMPF_HAVE_CUPTI
                           << "  -M <path>  Enable CUPTI memory monitoring and save CSV "
                              "files with given path prefix. For example, /tmp/test will "
                              "write files to /tmp/test_<rank>.csv (default: disabled)\n"
#endif
                           << "  -h         Display this help message\n";
                        if (rank == 0) {
                            std::cerr << ss.str();
                        }
                        if (use_mpi) {
                            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                        } else {
                            std::exit(0);
                        }
                    }
                    break;
                case 'C':
                    comm_type = std::string{optarg};
                    if (!(comm_type == "mpi" || comm_type == "ucxx")) {
                        if (rank == 0) {
                            std::cerr << "-C (Communicator) must be one of {mpi, ucxx}"
                                      << std::endl;
                        }
                        if (use_mpi) {
                            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                        } else {
                            std::exit(-1);
                        }
                    }
                    break;
                case 'O':
                    operation = std::string{optarg};
                    if (operation != "all-to-all") {
                        throw std::invalid_argument(
                            "-O (Operation) must be one of {all-to-all}"
                        );
                    }
                    break;
                case 'n':
                    parse_integer(msg_size, optarg);
                    break;
                case 'p':
                    parse_integer(num_ops, optarg);
                    break;
                case 'm':
                    rmm_mr = std::string{optarg};
                    if (!(rmm_mr == "cuda" || rmm_mr == "pool" || rmm_mr == "async"
                          || rmm_mr == "managed"))
                    {
                        throw std::invalid_argument(
                            "-m (RMM memory resource) must be one of {cuda, pool, async, "
                            "managed}"
                        );
                    }
                    break;
                case 'r':
                    parse_integer(num_runs, optarg);
                    break;
                case 'w':
                    parse_integer(num_warmups, optarg);
                    break;
#ifdef RAPIDSMPF_HAVE_CUPTI
                case 'M':
                    cupti_csv_prefix = std::string{optarg};
                    enable_cupti_monitoring = true;
                    break;
#endif
                case '?':
                    if (use_mpi) {
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                    } else {
                        std::exit(-1);
                    }
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
            if (use_mpi) {
                RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            } else {
                std::exit(-1);
            }
        }

        if (rmm_mr == "cuda") {
            if (rank == 0) {
                std::cout << "WARNING: using the default cuda memory resource "
                             "(-m cuda) might leak memory! A limitation in UCX "
                             "means that device memory send through IPC can "
                             "never be freed."
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
        ss << "  -C " << comm_type << " (communicator)\n";
        ss << "  -O " << operation << " (operation)\n";
        ss << "  -n " << msg_size << " (message size)\n";
        ss << "  -p " << num_ops << " (number of operations)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        if (enable_cupti_monitoring) {
            ss << "  -M " << cupti_csv_prefix << " (CUPTI memory monitoring enabled)\n";
        }
        comm.logger().print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::string rmm_mr{"pool"};
    std::string comm_type{"mpi"};
    std::string operation{"all-to-all"};
    std::uint64_t msg_size{1 << 20};
    std::uint64_t num_ops{1};
    bool enable_cupti_monitoring{false};
    std::string cupti_csv_prefix;
};

Duration run(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<rapidsmpf::Statistics> statistics
) {
    // Allocate send and recv buffers and fill the send buffers with random data.
    std::vector<std::unique_ptr<Buffer>> send_bufs;
    std::vector<std::unique_ptr<Buffer>> recv_bufs;
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < comm->nranks(); ++rank) {
            auto [res, _] = br->reserve(MemoryType::DEVICE, args.msg_size * 2, true);
            auto buf = br->allocate(args.msg_size, stream, res);
            random_fill(*buf, br->device_mr());
            send_bufs.push_back(std::move(buf));
            recv_bufs.push_back(br->allocate(args.msg_size, stream, res));
        }
    }

    // Sync before we start the timer.
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());

    auto const t0_elapsed = Clock::now();

    Tag const tag{0, 1};
    std::vector<std::unique_ptr<Communicator::Future>> futures;
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < static_cast<Rank>(comm->nranks()); ++rank) {
            auto buf = std::move(recv_bufs.at(
                static_cast<std::uint64_t>(rank)
                + i * static_cast<std::uint64_t>(comm->nranks())
            ));
            if (rank != comm->rank()) {
                statistics->add_bytes_stat("all-to-all-recv", buf->size);
                futures.push_back(comm->recv(rank, tag, std::move(buf)));
            }
        }
        for (Rank rank = 0; rank < static_cast<Rank>(comm->nranks()); ++rank) {
            auto buf = std::move(send_bufs.at(
                static_cast<std::uint64_t>(rank)
                + i * static_cast<std::uint64_t>(comm->nranks())
            ));
            if (rank != comm->rank()) {
                statistics->add_bytes_stat("all-to-all-send", buf->size);
                futures.push_back(comm->send(std::move(buf), rank, tag));
            }
        }
    }

    while (!futures.empty()) {
        std::ignore = comm->test_some(futures);
    }

    return Clock::now() - t0_elapsed;
}

int main(int argc, char** argv) {
    bool use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();

    int provided = 0;
    if (!use_bootstrap) {
        // Explicitly initialize MPI with thread support, as this is needed for both mpi
        // and ucxx communicators.
        RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

        RAPIDSMPF_EXPECTS(
            provided == MPI_THREAD_MULTIPLE,
            "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
        );
    }

    ArgumentParser args{argc, argv, !use_bootstrap};

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        if (use_bootstrap) {
            std::cerr << "Error: MPI communicator requires MPI initialization. "
                      << "Don't use with rrun or unset RAPIDSMPF_RANK." << std::endl;
            return 1;
        }
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, options);
    } else if (args.comm_type == "ucxx") {
        if (use_bootstrap) {
            // Launched with rrun - use bootstrap backend
            comm = rapidsmpf::bootstrap::create_ucxx_comm(
                rapidsmpf::bootstrap::Backend::AUTO, options
            );
        } else {
            // Launched with mpirun - use MPI bootstrap
            comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options);
        }
    } else {
        std::cerr << "Error: Unknown communicator type: " << args.comm_type << std::endl;
        return 1;
    }

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    args.pprint(*comm);
    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    BufferResource br{mr};

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
    auto stats = std::make_shared<rapidsmpf::Statistics>(/* enable = */ false);

#ifdef RAPIDSMPF_HAVE_CUPTI
    // Create CUPTI monitor if enabled
    std::unique_ptr<rapidsmpf::CuptiMonitor> cupti_monitor;
    if (args.enable_cupti_monitoring) {
        cupti_monitor = std::make_unique<rapidsmpf::CuptiMonitor>();
        cupti_monitor->start_monitoring();
        log.print("CUPTI memory monitoring enabled");
    }
#endif

    auto const local_messages_send =
        args.msg_size * args.num_ops * (static_cast<std::uint64_t>(comm->nranks()) - 1);
    auto const local_messages =
        args.msg_size * args.num_ops * static_cast<std::uint64_t>(comm->nranks());
    std::vector<double> elapsed_vec;
    for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
        // Enable statistics for the last run.
        if (i == args.num_warmups + args.num_runs - 1) {
            stats = std::make_shared<rapidsmpf::Statistics>();
        }
        auto const elapsed = run(comm, args, stream, &br, stats).count();
        std::stringstream ss;
        ss << "elapsed: " << to_precision(elapsed) << " sec"
           << " | local comm: " << format_nbytes(local_messages_send / elapsed)
           << "/s | local throughput: " << format_nbytes(local_messages / elapsed)
           << "/s | global throughput: "
           << format_nbytes(
                  local_messages * static_cast<std::uint64_t>(comm->nranks()) / elapsed
              )
           << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log.print(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }
    log.print(stats->report("Statistics (of the last run):"));

#ifdef RAPIDSMPF_HAVE_CUPTI
    // Save CUPTI monitoring results to CSV file
    if (args.enable_cupti_monitoring && cupti_monitor) {
        cupti_monitor->stop_monitoring();

        std::string csv_filename =
            args.cupti_csv_prefix + std::to_string(comm->rank()) + ".csv";
        try {
            cupti_monitor->write_csv(csv_filename);
            log.print(
                "CUPTI memory data written to " + csv_filename + " ("
                + std::to_string(cupti_monitor->get_sample_count()) + " samples, "
                + std::to_string(cupti_monitor->get_total_callback_count())
                + " callbacks)"
            );

            // Print callback summary for rank 0
            if (comm->rank() == 0) {
                log.print(
                    "CUPTI Callback Summary:\n" + cupti_monitor->get_callback_summary()
                );
            }
        } catch (std::exception const& e) {
            log.print("Failed to write CUPTI CSV file: " + std::string(e.what()));
        }
    }
#endif

    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Finalize());
    }
    return 0;
}
