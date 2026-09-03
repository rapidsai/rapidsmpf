/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/string.hpp>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

#include "utils/misc.hpp"
#include "utils/rmm_utils.hpp"

using namespace rapidsmpf;
using rapidsmpf::shuffler::PartID;
using rapidsmpf::shuffler::Shuffler;

namespace {

/**
 * @brief Compact metadata describing a pre-partitioned shuffle chunk.
 */
struct ChunkHeader {
    Rank src_rank{};
    std::uint32_t batch{};
    PartID dest_partition{};
    std::uint64_t payload_size{};
    std::uint8_t fill_byte{};

    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> to_metadata() const {
        auto const* bytes = reinterpret_cast<std::uint8_t const*>(this);
        return std::make_unique<std::vector<std::uint8_t>>(bytes, bytes + sizeof(*this));
    }

    [[nodiscard]] static ChunkHeader from_metadata(
        std::vector<std::uint8_t> const& metadata
    ) {
        RAPIDSMPF_EXPECTS(
            metadata.size() == sizeof(ChunkHeader),
            "unexpected metadata size",
            std::runtime_error
        );
        ChunkHeader header{};
        std::memcpy(&header, metadata.data(), sizeof(header));
        return header;
    }
};

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv, bool use_mpi = true) {
        int rank = 0;

        if (use_mpi) {
            RAPIDSMPF_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");
            RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        } else {
            std::ignore = rapidsmpf::bootstrap::get_nranks();
        }

        try {
            int option;
            while ((option = getopt(argc, argv, "hC:r:w:n:p:o:m:M:s")) != -1) {
                switch (option) {
                case 'h':
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                           << "             ucxx automatically detects launcher (mpirun "
                              "or rrun)\n"
                           << "  -n <num>   Payload size in bytes per partition chunk "
                              "(default: 1M)\n"
                           << "  -p <num>   Number of insertion batches per rank "
                              "(default: 1)\n"
                           << "  -o <num>   Output partitions per rank (default: 1)\n"
                           << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                              "managed} "
                              "(default: pool)\n"
                           << "  -r <num>   Number of runs (default: 1)\n"
                           << "  -w <num>   Number of warmup runs (default: 0)\n"
                           << "  -s         Discard extracted output (skip validation)\n"
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
                case 'n':
                    parse_integer(payload_size, optarg, 1);
                    break;
                case 'p':
                    parse_integer(num_batches, optarg, 1);
                    break;
                case 'o':
                    parse_integer(output_partitions_per_rank, optarg, 1);
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
                case 's':
                    discard_output = true;
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
        ss << "  -n " << payload_size << " (payload size)\n";
        ss << "  -p " << num_batches << " (insertion batches per rank)\n";
        ss << "  -o " << output_partitions_per_rank << " (output partitions per rank)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        ss << "  -s " << (discard_output ? "true" : "false") << " (discard output)\n";
        if (enable_cupti_monitoring) {
            ss << "  -M " << cupti_csv_prefix << " (CUPTI memory monitoring enabled)\n";
        }
        comm.logger()->print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::string rmm_mr{"pool"};
    std::string comm_type{"mpi"};
    std::uint64_t payload_size{1 << 20};
    std::uint64_t num_batches{1};
    std::uint64_t output_partitions_per_rank{1};
    bool discard_output{false};
    bool enable_cupti_monitoring{false};
    std::string cupti_csv_prefix;
};

void comm_barrier(std::shared_ptr<Communicator> const& comm, bool mpi_initialized) {
    if (auto ucxx_comm = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm)) {
        ucxx_comm->barrier();
    } else if (mpi_initialized) {
        RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));
    }
}

[[nodiscard]] std::uint8_t chunk_fill_byte(
    Rank src_rank, std::uint64_t batch, PartID dest_partition
) {
    return static_cast<std::uint8_t>(
        (safe_cast<std::uint64_t>(src_rank) + batch
         + static_cast<std::uint64_t>(dest_partition))
        & 0xFF
    );
}

[[nodiscard]] PackedData make_chunk(
    Rank src_rank,
    std::uint64_t batch,
    PartID dest_partition,
    std::uint64_t size,
    cuda::stream_ref stream,
    BufferResource& br
) {
    ChunkHeader const header{
        src_rank,
        static_cast<std::uint32_t>(batch),
        dest_partition,
        size,
        chunk_fill_byte(src_rank, batch, dest_partition),
    };
    auto metadata = header.to_metadata();
    auto const* fill_byte = metadata->data() + offsetof(ChunkHeader, fill_byte);
    auto [reservation, _] = br.reserve(MemoryType::DEVICE, size, AllowOverbooking::YES);
    auto data = br.make_buffer(stream, std::move(reservation));
    data->write_access([fill_byte, size](std::byte* ptr, cuda::stream_ref op_stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, *fill_byte, size, op_stream.get()));
    });
    return PackedData{std::move(metadata), std::move(data)};
}

[[nodiscard]] std::vector<std::unordered_map<PartID, PackedData>> generate_batches(
    Communicator const& comm,
    ArgumentParser const& args,
    PartID total_num_partitions,
    cuda::stream_ref stream,
    BufferResource& br
) {
    std::vector<std::unordered_map<PartID, PackedData>> batches;
    batches.reserve(args.num_batches);
    for (std::uint64_t batch = 0; batch < args.num_batches; ++batch) {
        std::unordered_map<PartID, PackedData> chunks;
        chunks.reserve(total_num_partitions);
        for (PartID pid = 0; pid < total_num_partitions; ++pid) {
            chunks.emplace(
                pid, make_chunk(comm.rank(), batch, pid, args.payload_size, stream, br)
            );
        }
        batches.push_back(std::move(chunks));
    }
    return batches;
}

void validate_chunk(
    PackedData&& packed,
    Rank expected_src_rank,
    std::uint64_t expected_batch,
    PartID expected_dest_partition,
    std::uint64_t expected_payload_size
) {
    auto const header = ChunkHeader::from_metadata(*packed.metadata);
    RAPIDSMPF_EXPECTS(
        header.src_rank == expected_src_rank,
        "unexpected source rank in metadata",
        std::runtime_error
    );
    RAPIDSMPF_EXPECTS(
        header.batch == expected_batch, "unexpected batch in metadata", std::runtime_error
    );
    RAPIDSMPF_EXPECTS(
        header.dest_partition == expected_dest_partition,
        "unexpected destination partition in metadata",
        std::runtime_error
    );
    RAPIDSMPF_EXPECTS(
        header.payload_size == expected_payload_size,
        "unexpected payload size in metadata",
        std::runtime_error
    );
    RAPIDSMPF_EXPECTS(
        packed.data->size == expected_payload_size,
        "unexpected buffer size",
        std::runtime_error
    );

    auto const expected_fill =
        chunk_fill_byte(expected_src_rank, expected_batch, expected_dest_partition);
    RAPIDSMPF_EXPECTS(
        header.fill_byte == expected_fill,
        "unexpected fill byte in metadata",
        std::runtime_error
    );

    packed.data->latest_write_event().host_wait();
    std::vector<std::uint8_t> bytes(expected_payload_size);
    if (contains(Buffer::host_buffer_types, packed.data->mem_type())) {
        std::memcpy(bytes.data(), packed.data->data(), expected_payload_size);
    } else {
        RAPIDSMPF_CUDA_TRY(cudaMemcpy(
            bytes.data(),
            packed.data->data(),
            expected_payload_size,
            cudaMemcpyDeviceToHost
        ));
    }
    for (std::uint8_t byte : bytes) {
        RAPIDSMPF_EXPECTS(
            byte == expected_fill, "unexpected payload byte", std::runtime_error
        );
    }
}

void validate_extracted(
    std::unordered_map<PartID, std::vector<PackedData>>& extracted,
    std::shared_ptr<Communicator> const& comm,
    ArgumentParser const& args,
    PartID total_num_partitions
) {
    auto const local_partitions =
        Shuffler::local_partitions(comm, total_num_partitions, Shuffler::round_robin);
    RAPIDSMPF_EXPECTS(
        extracted.size() == local_partitions.size(),
        "unexpected number of extracted partitions",
        std::runtime_error
    );

    auto const expected_chunks_per_partition =
        args.num_batches * static_cast<std::uint64_t>(comm->nranks());

    for (PartID const pid : local_partitions) {
        auto it = extracted.find(pid);
        RAPIDSMPF_EXPECTS(
            it != extracted.end(), "missing extracted partition", std::runtime_error
        );
        RAPIDSMPF_EXPECTS(
            it->second.size() == expected_chunks_per_partition,
            "unexpected chunk count for partition",
            std::runtime_error
        );

        std::vector<bool> seen(
            args.num_batches * static_cast<std::size_t>(comm->nranks()), false
        );
        for (auto& packed : it->second) {
            auto const header = ChunkHeader::from_metadata(*packed.metadata);
            RAPIDSMPF_EXPECTS(
                header.dest_partition == pid,
                "chunk routed to wrong partition",
                std::runtime_error
            );
            auto const key = static_cast<std::size_t>(header.batch)
                                 * static_cast<std::size_t>(comm->nranks())
                             + static_cast<std::size_t>(header.src_rank);
            RAPIDSMPF_EXPECTS(
                key < seen.size() && !seen[key],
                "duplicate source/batch chunk",
                std::runtime_error
            );
            seen[key] = true;
            validate_chunk(
                std::move(packed), header.src_rank, header.batch, pid, args.payload_size
            );
        }
    }
}

Duration run_shuffle(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    PartID total_num_partitions,
    BufferResource& br,
    std::vector<std::unordered_map<PartID, PackedData>>& batches
) {
    Shuffler shuffler(comm, 0, total_num_partitions, &br);

    comm_barrier(comm, mpi::is_initialized());
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto const t0_elapsed = Clock::now();

    for (auto& batch : batches) {
        shuffler.insert(std::move(batch));
    }
    shuffler.insert_finished();
    shuffler.wait(std::chrono::seconds{3600});

    std::unordered_map<PartID, std::vector<PackedData>> extracted;
    if (!args.discard_output) {
        for (PartID const pid : shuffler.local_partitions()) {
            extracted.emplace(pid, shuffler.extract(pid));
        }
    } else {
        for (PartID const pid : shuffler.local_partitions()) {
            std::ignore = shuffler.extract(pid);
        }
    }

    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto const elapsed = Clock::now() - t0_elapsed;

    if (!args.discard_output) {
        validate_extracted(extracted, comm, args, total_num_partitions);
    }

    return elapsed;
}

}  // namespace

int main(int argc, char** argv) {
    bool const use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();

    int provided = 0;
    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
        RAPIDSMPF_EXPECTS(
            provided == MPI_THREAD_MULTIPLE,
            "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
        );
    }

    ArgumentParser args{argc, argv, !use_bootstrap};

    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    auto stats = rapidsmpf::Statistics::disabled();
    auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>(stats);
    auto logger = rapidsmpf::Logger::from_options(options);
    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        if (use_bootstrap) {
            std::cerr << "Error: MPI communicator requires MPI initialization. "
                      << "Don't use with rrun or unset RRUN_RANK." << std::endl;
            return 1;
        }
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, progress_thread, logger);
    } else if (args.comm_type == "ucxx") {
        if (use_bootstrap) {
            comm = rapidsmpf::bootstrap::create_ucxx_comm(
                progress_thread, rapidsmpf::bootstrap::BackendType::AUTO, options, logger
            );
        } else {
            comm = rapidsmpf::ucxx::init_using_mpi(
                MPI_COMM_WORLD, options, progress_thread, logger
            );
        }
    } else {
        std::cerr << "Error: Unknown communicator type: " << args.comm_type << std::endl;
        return 1;
    }

    auto& log = comm->logger();
    cuda::stream_ref stream{cudaStreamLegacy};
    args.pprint(*comm);
    set_current_rmm_resource(args.rmm_mr);

    auto br = BufferResource::from_options(
        rmm::mr::get_current_device_resource_ref(), options, stats
    );

    auto const total_num_partitions = safe_cast<PartID>(
        args.output_partitions_per_rank * static_cast<std::uint64_t>(comm->nranks())
    );

    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');
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
        ss << "  Total partitions: " << total_num_partitions << "\n";
        ss << "  Local partitions: " << args.output_partitions_per_rank << "\n";
        ss << "  BufferResource configured from environment options\n";
        log->print(ss.str());
    }

#ifdef RAPIDSMPF_HAVE_CUPTI
    std::unique_ptr<rapidsmpf::CuptiMonitor> cupti_monitor;
    if (args.enable_cupti_monitoring) {
        cupti_monitor = std::make_unique<rapidsmpf::CuptiMonitor>();
        cupti_monitor->start_monitoring();
        log->print("CUPTI memory monitoring enabled");
    }
#endif

    auto const nranks = static_cast<std::uint64_t>(comm->nranks());
    auto const local_logical_bytes =
        args.payload_size * args.num_batches * args.output_partitions_per_rank * nranks;
    auto const local_network_bytes = args.payload_size * args.num_batches
                                     * args.output_partitions_per_rank
                                     * (nranks > 0 ? nranks - 1 : 0);

    std::vector<double> elapsed_vec;
    for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
        if (i == args.num_warmups + args.num_runs - 1) {
            stats->enable();
        }

        auto batches = generate_batches(*comm, args, total_num_partitions, stream, *br);
        auto const elapsed =
            run_shuffle(comm, args, total_num_partitions, *br, batches).count();

        std::stringstream ss;
        ss << "elapsed: " << format_duration(elapsed);
        if (local_network_bytes > 0) {
            ss << " | local comm: " << format_nbytes(local_network_bytes / elapsed)
               << "/s";
        }
        ss << " | local throughput: " << format_nbytes(local_logical_bytes / elapsed)
           << "/s | global throughput: "
           << format_nbytes(local_logical_bytes * nranks / elapsed) << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log->print(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }

    {
        auto const elapsed_mean = harmonic_mean(elapsed_vec);
        std::stringstream ss;
        ss << "means: " << format_duration(elapsed_mean);
        if (local_network_bytes > 0) {
            ss << " | local comm: " << format_nbytes(local_network_bytes / elapsed_mean)
               << "/s";
        }
        ss << " | local throughput: " << format_nbytes(local_logical_bytes / elapsed_mean)
           << "/s | global throughput: "
           << format_nbytes(local_logical_bytes * nranks / elapsed_mean)
           << "/s | num_batches: " << args.num_batches
           << " | output_partitions_per_rank: " << args.output_partitions_per_rank
           << " | nranks: " << comm->nranks();
        log->print(ss.str());
    }
    log->print(stats->report({.header = "Statistics (of the last run):"}));

#ifdef RAPIDSMPF_HAVE_CUPTI
    if (args.enable_cupti_monitoring && cupti_monitor) {
        cupti_monitor->stop_monitoring();

        std::string csv_filename =
            args.cupti_csv_prefix + std::to_string(comm->rank()) + ".csv";
        try {
            cupti_monitor->write_csv(csv_filename);
            log->print(
                "CUPTI memory data written to " + csv_filename + " ("
                + std::to_string(cupti_monitor->get_sample_count()) + " samples, "
                + std::to_string(cupti_monitor->get_total_callback_count())
                + " callbacks)"
            );

            if (comm->rank() == 0) {
                log->print(
                    "CUPTI Callback Summary:\n" + cupti_monitor->get_callback_summary()
                );
            }
        } catch (std::exception const& e) {
            log->print("Failed to write CUPTI CSV file: " + std::string(e.what()));
        }
    }
#endif

    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Finalize());
    }
    return 0;
}
