/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>

#include <mpi.h>
#include <ucxx/request_am.h>
#include <ucxx/typedefs.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>

#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_stack.hpp"


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
            while ((option = getopt(argc, argv, "hC:O:A:r:w:n:p:m:")) != -1) {
                switch (option) {
                case 'h':
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                           << "  -O <op>    Operation {all-to-all} (default: "
                              "all-to-all)\n"
                           << "  -A <api>   API type {tag, am} (default: tag\n"
                           << "  -n <num>   Message size in bytes (default: 1M)\n"
                           << "  -p <num>   Number of concurrent operations, e.g. number"
                              " of  concurrent all-to-all operations (default: 1)\n"
                           << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                              "managed} "
                              "(default: cuda)\n"
                           << "  -r <num>   Number of runs (default: 1)\n"
                           << "  -w <num>   Number of warmup runs (default: 0)\n"
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
                case 'O':
                    operation = std::string{optarg};
                    if (operation != "all-to-all") {
                        throw std::invalid_argument(
                            "-O (Operation) must be one of {all-to-all}"
                        );
                    }
                    break;
                case 'A':
                    api_type = std::string{optarg};
                    if (!(api_type == "tag" || api_type == "am")) {
                        if (rank == 0) {
                            std::cerr << "-A (API type) must be one of {tag, am}"
                                      << std::endl;
                        }
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
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

        if (api_type == "am" && comm_type != "ucxx") {
            std::cerr << "'-A am' is only supported with '-C ucxx'" << std::endl;
            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
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
        ss << "  -A " << api_type << " (API type)\n";
        ss << "  -n " << msg_size << " (message size)\n";
        ss << "  -p " << num_ops << " (number of operations)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        comm.logger().print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::string rmm_mr{"cuda"};
    std::string comm_type{"mpi"};
    std::string operation{"all-to-all"};
    std::string api_type{"tag"};
    std::uint64_t msg_size{1 << 20};
    std::uint64_t num_ops{1};
};

void run_tag(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    std::shared_ptr<rapidsmpf::Statistics> statistics,
    std::vector<std::unique_ptr<Buffer>>& send_bufs,
    std::vector<std::unique_ptr<Buffer>>& recv_bufs
) {
    Tag const tag{0, 1};
    std::vector<std::unique_ptr<Communicator::Future>> futures;
    futures.reserve(send_bufs.size() + recv_bufs.size());
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < static_cast<Rank>(comm->nranks()); ++rank) {
            auto buf = std::move(recv_bufs.at(
                static_cast<std::uint64_t>(rank)
                + i * static_cast<std::uint64_t>(comm->nranks())
            ));
            if (rank != comm->rank()) {
                statistics->add_bytes_stat("all-to-all-recv", buf->size);
                futures.emplace_back(comm->recv(rank, tag, std::move(buf)));
            }
        }
        for (Rank rank = 0; rank < static_cast<Rank>(comm->nranks()); ++rank) {
            auto buf = std::move(send_bufs.at(
                static_cast<std::uint64_t>(rank)
                + i * static_cast<std::uint64_t>(comm->nranks())
            ));
            if (rank != comm->rank()) {
                statistics->add_bytes_stat("all-to-all-send", buf->size);
                futures.emplace_back(comm->send(std::move(buf), rank, tag));
            }
        }
    }

    while (!futures.empty()) {
        std::ignore = comm->test_some(futures);
    }
}

struct AmHeader {
    std::uint64_t op_num_;
    Rank rank_;
    std::uint64_t iteration_id_;  // Add iteration ID to prevent cross-iteration races

    AmHeader(std::uint64_t op_num, Rank rank, std::uint64_t iteration_id = 0)
        : op_num_(op_num), rank_(rank), iteration_id_(iteration_id) {}

    [[nodiscard]] std::uint64_t buf_index(Rank nranks) const {
        return static_cast<std::uint64_t>(rank_)
               + op_num_ * static_cast<std::uint64_t>(nranks);
    }
};

struct PendingMessage {
    std::shared_ptr<::ucxx::Request> req;
    AmHeader header;

    PendingMessage(std::shared_ptr<::ucxx::Request> req, AmHeader header)
        : req(std::move(req)), header(header) {}
};

class AmCallbackContainer {
  private:
    std::vector<std::unique_ptr<Buffer>> recv_bufs_{};
    std::vector<std::unique_ptr<Communicator::Future>> recv_futures_{};
    std::mutex recv_mutex_{};
    size_t recv_count_{0};
    std::shared_ptr<rapidsmpf::ucxx::UCXX> comm_{nullptr};
    std::atomic<std::uint64_t> current_iteration_id_{0};
    std::queue<PendingMessage> pending_messages_{};

    void process_message(
        std::shared_ptr<::ucxx::Request> req, const AmHeader& am_header
    ) {
        std::uint64_t buf_index =
            static_cast<std::uint64_t>(am_header.rank_)
            + am_header.op_num_ * static_cast<std::uint64_t>(comm_->nranks());

        // Bounds check to prevent crashes
        if (buf_index >= recv_bufs_.size() || !recv_bufs_[buf_index]) {
            std::cerr << "ERROR: Invalid buffer index " << buf_index
                      << " for recv_bufs_ of size " << recv_bufs_.size()
                      << " (rank=" << am_header.rank_ << ", op=" << am_header.op_num_
                      << ", iter=" << am_header.iteration_id_ << ")" << std::endl;
            return;
        }

        auto future = comm_->am_recv(
            std::dynamic_pointer_cast<::ucxx::RequestAm>(req),
            std::move(recv_bufs_[buf_index])
        );
        auto lock = acquire_lock();
        recv_futures_.push_back(std::move(future));
        ++recv_count_;
    }

  public:
    AmCallbackContainer(std::shared_ptr<rapidsmpf::ucxx::UCXX> comm)
        : comm_(std::move(comm)) {}

    [[nodiscard]] std::lock_guard<std::mutex> acquire_lock() {
        return std::lock_guard<std::mutex>(recv_mutex_);
    }

    [[nodiscard]] std::vector<std::unique_ptr<Communicator::Future>> release_futures() {
        return std::move(recv_futures_);
    }

    [[nodiscard]] size_t get_recv_count() {
        return recv_count_;
    }

    void reset(
        std::vector<std::unique_ptr<Buffer>>&& recv_bufs, std::uint64_t iteration_id
    ) {
        std::vector<PendingMessage> messages_to_process;

        {
            auto lock = acquire_lock();
            recv_bufs_ = std::move(recv_bufs);
            recv_futures_.clear();
            recv_count_ = 0;
            current_iteration_id_.store(iteration_id, std::memory_order_release);
        }

        process_ready_pending_messages();
    }

    [[nodiscard]] std::uint64_t get_current_iteration() const {
        return current_iteration_id_.load(std::memory_order_acquire);
    }

    // Process any pending messages that are now ready for the current iteration
    void process_ready_pending_messages() {
        std::uint64_t current_iteration = get_current_iteration();
        std::queue<PendingMessage> remaining_messages;

        while (!pending_messages_.empty()) {
            auto& pending = pending_messages_.front();
            if (pending.header.iteration_id_ == current_iteration) {
                // Process this message now
                process_message(pending.req, pending.header);
            } else {
                // Keep for future iterations
                remaining_messages.push(std::move(pending));
            }
            pending_messages_.pop();
        }

        // Replace queue with remaining messages
        pending_messages_ = std::move(remaining_messages);
    }

    void enqueue_or_process_message(
        std::shared_ptr<::ucxx::Request> req, const AmHeader& header
    ) {
        std::uint64_t current_iteration = get_current_iteration();

        if (header.iteration_id_ == current_iteration) {
            // Process immediately - recv_bufs_ should be available for current iteration
            process_message(req, header);
        } else if (header.iteration_id_ > current_iteration) {
            // Queue for future iteration
            pending_messages_.emplace(req, header);
        }
        // Note: We ignore messages from past iterations (header.iteration_id_ <
        // current_iteration) as they should not occur in normal operation
    }

    static std::unique_ptr<AmCallbackContainer> setup(
        std::shared_ptr<rapidsmpf::ucxx::UCXX> comm
    ) {
        auto container = std::make_unique<AmCallbackContainer>(comm);

        ::ucxx::AmReceiverCallbackInfo receiverCallbackInfo("RapidsMPF-bench-comm", 0);
        auto receiverCallback =
            ::ucxx::AmReceiverCallbackType([container_ptr = container.get()](
                                               std::shared_ptr<::ucxx::Request> req,
                                               ucp_ep_h /*ep*/,
                                               ::ucxx::AmReceiverCallbackInfo const& info
                                           ) {
                auto am_header = info.userHeader->as<AmHeader>();

                // Use the new queue-based approach that handles messages correctly
                // instead of dropping them
                container_ptr->enqueue_or_process_message(req, am_header);
            });
        comm->am_recv_callback(receiverCallbackInfo, receiverCallback);

        return container;
    }
};

void run_am(
    std::shared_ptr<rapidsmpf::ucxx::UCXX> comm,
    ArgumentParser const& args,
    std::shared_ptr<rapidsmpf::Statistics> statistics,
    std::vector<std::unique_ptr<Buffer>>& send_bufs,
    std::shared_ptr<AmCallbackContainer> am_callback_container,
    std::uint64_t iteration_id = 0
) {
    std::vector<std::unique_ptr<Communicator::Future>> send_futures;
    send_futures.reserve(send_bufs.size());
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < static_cast<Rank>(comm->nranks()); ++rank) {
            AmHeader am_header(i, comm->rank(), iteration_id);
            auto buf = std::move(send_bufs.at(
                static_cast<std::uint64_t>(rank)
                + i * static_cast<std::uint64_t>(comm->nranks())
            ));

            if (rank != comm->rank()) {
                statistics->add_bytes_stat("all-to-all-send", buf->size);

                ::ucxx::AmReceiverCallbackInfo info(
                    "RapidsMPF-bench-comm", 0, true, ::ucxx::AmUserHeader(am_header)
                );
                send_futures.emplace_back(comm->am_send(std::move(buf), rank, info));
            }
        }
    }

    size_t recv_count = 0;
    std::vector<std::unique_ptr<Communicator::Future>> recv_futures{};
    {
        // Calculate expected receive count for this iteration:
        // num_ops operations * (nranks - 1) senders per operation
        std::uint64_t expected_recv_count =
            args.num_ops * (static_cast<std::uint64_t>(comm->nranks()) - 1);

        while (!send_futures.empty() || !recv_futures.empty()
               || recv_count < expected_recv_count)
        {
            if (!send_futures.empty())
                std::ignore = comm->test_some(send_futures);

            {
                // This block prevents holding the lock to test for completion/receive
                // data
                auto lock = am_callback_container->acquire_lock();

                auto callback_futures = am_callback_container->release_futures();
                recv_futures.insert(
                    recv_futures.end(),
                    std::make_move_iterator(callback_futures.begin()),
                    std::make_move_iterator(callback_futures.end())
                );
                recv_count = am_callback_container->get_recv_count();
            }

            if (!recv_futures.empty())
                std::ignore = comm->test_some(recv_futures);

            // Process any pending messages that are now ready for this iteration
            am_callback_container->process_ready_pending_messages();
        }
    }

    for (size_t i = 0; i < send_futures.size(); i++) {
        auto ucxx_future =
            dynamic_cast<rapidsmpf::ucxx::UCXX::Future const*>(send_futures[i].get());
        RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    }
    for (size_t i = 0; i < recv_futures.size(); i++) {
        auto ucxx_future =
            dynamic_cast<rapidsmpf::ucxx::UCXX::Future const*>(recv_futures[i].get());
        RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    }
}

Duration run(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<rapidsmpf::Statistics> statistics,
    std::shared_ptr<AmCallbackContainer> am_callback_container = nullptr,
    std::uint64_t iteration_id = 0
) {
    // Allocate send and recv buffers and fill the send buffers with random data.
    std::vector<std::unique_ptr<Buffer>> send_bufs;
    std::vector<std::unique_ptr<Buffer>> recv_bufs;
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < comm->nranks(); ++rank) {
            auto [res, _] = br->reserve(MemoryType::DEVICE, args.msg_size * 2, true);
            auto buf = br->allocate(MemoryType::DEVICE, args.msg_size, stream, res);
            random_fill(*buf, stream, br->device_mr());
            send_bufs.push_back(std::move(buf));
            recv_bufs.push_back(
                br->allocate(MemoryType::DEVICE, args.msg_size, stream, res)
            );
        }
    }

    if (am_callback_container != nullptr) {
        am_callback_container->reset(std::move(recv_bufs), iteration_id);

        // Required to ensure all workers have setup recv buffers before starting
        std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm)->barrier();
    }

    auto const t0_elapsed = Clock::now();

    if (args.api_type == "tag") {
        run_tag(comm, args, statistics, send_bufs, recv_bufs);
    } else if (args.api_type == "am") {
        RAPIDSMPF_EXPECTS(
            am_callback_container != nullptr,
            "AM callback container is required for AM API"
        );
        run_am(
            std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm),
            args,
            statistics,
            send_bufs,
            am_callback_container,
            iteration_id
        );

        // No barrier needed here anymore! The iteration-based message queuing
        // prevents race conditions between iterations by queuing future messages
        // instead of dropping them.
    }

    return Clock::now() - t0_elapsed;
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

    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, options);
    } else {  // ucxx
        comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options);
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

    // Setup AM callback container once before the loop if using AM API
    std::shared_ptr<AmCallbackContainer> am_callback_container = nullptr;
    if (args.api_type == "am") {
        am_callback_container = AmCallbackContainer::setup(
            std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm)
        );
    }

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
        auto const elapsed =
            run(comm, args, stream, &br, stats, am_callback_container, i).count();
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
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
