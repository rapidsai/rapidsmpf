/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/communicator/ucxx_utils.hpp>

#include "utils/misc.hpp"
#include "utils/random_data.hpp"


using namespace rapidsmp;

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv) {
        RAPIDSMP_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");

        int rank, nranks;
        RAPIDSMP_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        RAPIDSMP_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

        int option;
        while ((option = getopt(argc, argv, "hC:r:w:n:")) != -1) {
            switch (option) {
            case 'h':
                {
                    std::stringstream ss;
                    ss << "Usage: " << argv[0] << " [options]\n"
                       << "Options:\n"
                       << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                       << "  -r <num>   Number of runs (default: 1)\n"
                       << "  -w <num>   Number of warmup runs (default: 0)\n"
                       << "  -n <num>   Message size in bytes (default: 1M)\n"
                       << "  -h         Display this help message\n";
                    if (rank == 0) {
                        std::cerr << ss.str();
                    }
                    RAPIDSMP_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                }
            case 'C':
                comm_type = std::string{optarg};
                if (!(comm_type == "mpi" || comm_type == "ucxx")) {
                    if (rank == 0) {
                        std::cerr << "-C (Communicator) must be one of {mpi, ucxx}"
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
            case 'n':
                msg_size = std::stoull(optarg);
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
    }

    void pprint(Communicator& comm) const {
        if (comm.rank() > 0) {
            return;
        }
        std::stringstream ss;
        ss << "Arguments:\n";
        ss << "  -c " << comm_type << " (communicator)\n";
        ss << "  -n " << msg_size << " (message size)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        comm.logger().info(ss.str());
    }

    int num_runs{1};
    int num_warmups{0};
    std::string comm_type{"mpi"};
    std::uint64_t msg_size{1 << 20};
};

Duration run(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    // Allocate send and recv buffers and fill the send buffers with random data.
    std::vector<std::unique_ptr<Buffer>> send_bufs;
    std::vector<std::unique_ptr<Buffer>> recv_bufs;
    for (Rank rank = 0; rank < comm->nranks(); ++rank) {
        auto [reservation, _] = br->reserve(MemoryType::DEVICE, args.msg_size * 2, true);
        auto buf = br->allocate(MemoryType::DEVICE, args.msg_size, stream, reservation);
        random_fill(*buf, stream, br->device_mr());
        send_bufs.push_back(std::move(buf));
        recv_bufs.push_back(
            br->allocate(MemoryType::DEVICE, args.msg_size, stream, reservation)
        );
    }

    auto const t0_elapsed = Clock::now();

    Tag const tag{0, 1};
    std::vector<std::unique_ptr<Communicator::Future>> futures;
    for (Rank rank = 0; rank < comm->nranks(); ++rank) {
        if (rank != comm->rank()) {
            futures.push_back(comm->recv(rank, tag, std::move(recv_bufs.at(rank)), stream)
            );
        }
    }
    for (Rank rank = 0; rank < comm->nranks(); ++rank) {
        if (rank != comm->rank()) {
            futures.push_back(comm->send(std::move(send_bufs.at(rank)), rank, tag, stream)
            );
        }
    }

    while (!futures.empty()) {
        std::vector<std::size_t> finished = comm->test_some(futures);
        // Sort the indexes into descending order.
        std::sort(finished.begin(), finished.end(), std::greater<>());
        // And erase from the right.
        for (auto i : finished) {
            futures.erase(futures.begin() + i);
        }
    }

    auto const t1_elapsed = Clock::now();
    return t1_elapsed - t0_elapsed;
}

int main(int argc, char** argv) {
    // Explicitly initialize MPI with thread support, as this is needed for both mpi and
    // ucxx communicators.
    int provided;
    RAPIDSMP_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    RAPIDSMP_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );

    ArgumentParser args{argc, argv};

    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD);
    } else {  // ucxx
        comm = rapidsmp::ucxx::init_using_mpi(MPI_COMM_WORLD);
    }

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    BufferResource br{mr};

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    args.pprint(*comm);

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
        ss << "    Total Memory: " << format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.info(ss.str());
    }

    std::vector<double> elapsed_vec;
    for (auto i = 0; i < args.num_warmups + args.num_runs; ++i) {
        auto const elapsed = run(comm, args, stream, &br).count();
        std::stringstream ss;
        ss << "elapsed: " << to_precision(elapsed) << " sec "
           << "| bandwidth: " << format_nbytes(args.msg_size / elapsed) << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log.info(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }
    RAPIDSMP_MPI(MPI_Finalize());
    return 0;
}
