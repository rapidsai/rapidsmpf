/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "utilities.hpp"

namespace {

[[maybe_unused]] rapidsmpf::streaming::Node read_parquet_chunk(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    rapidsmpf::streaming::ThrottlingAdaptor& ch_out,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto ticket = co_await ch_out.acquire();
    auto stream = ctx->br()->stream_pool().get_stream();
    auto tbl = cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl;
    auto [_, receipt] = co_await ticket.send(
        rapidsmpf::streaming::to_message(
            sequence_number,
            std::make_unique<rapidsmpf::streaming::TableChunk>(std::move(tbl), stream)
        )
    );
    co_await ctx->executor()->yield();
    co_await receipt;
}

[[maybe_unused]] rapidsmpf::streaming::Node read_parquet_unordered(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    cudf::size_type num_rows_per_chunk
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_out};
    auto throttle = rapidsmpf::streaming::ThrottlingAdaptor(
        ch_out, static_cast<std::ptrdiff_t>(num_producers)
    );

    std::vector<rapidsmpf::streaming::Node> tasks{};
    auto metadata = cudf::io::read_parquet_metadata(options.get_source());
    auto skip_rows = options.get_skip_rows();
    auto num_rows = metadata.num_rows();
    std::uint64_t sequence_number = 0;
    while (skip_rows < num_rows) {
        auto chunk_num_rows =
            std::min(static_cast<std::int64_t>(num_rows_per_chunk), num_rows - skip_rows);
        cudf::io::parquet_reader_options local_options{options};
        local_options.set_skip_rows(skip_rows);
        local_options.set_num_rows(chunk_num_rows);
        skip_rows += chunk_num_rows;
        tasks.push_back(
            read_parquet_chunk(ctx, throttle, local_options, sequence_number++)
        );
    }
    rapidsmpf::streaming::coro_results(co_await coro::when_all(std::move(tasks)));
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    bool is_dir,
    bool is_unordered
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        is_dir ? "lineitem" : "lineitem.parquet"
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns(
                           {"l_discount",
                            "l_extendedprice",
                            "l_orderkey",
                            "l_partkey",
                            "l_quantity",
                            "l_suppkey"}
                       )
                       .build();
    if (is_unordered) {
        return read_parquet_unordered(
            ctx, ch_out, num_producers, options, num_rows_per_chunk
        );
    } else {
        return rapidsmpf::streaming::node::read_parquet(
            ctx, ch_out, num_producers, options, num_rows_per_chunk
        );
    }
}

[[maybe_unused]] rapidsmpf::streaming::Node consume(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        ctx->comm()->logger().print(
            "Consumed chunk with ",
            chunk.table_view().num_rows(),
            " rows and ",
            chunk.table_view().num_columns(),
            " columns"
        );
    }
}

}  // namespace

int main(int argc, char** argv) {
    cudaFree(nullptr);
    rapidsmpf::mpi::init(&argc, &argv);
    MPI_Comm mpi_comm;
    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    rmm::mr::cuda_async_memory_resource mr{};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);
    rapidsmpf::BufferResource br(stats_mr);  // , std::move(memory_available));
    auto options =
        rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
    options.insert_if_absent("num_streaming_threads", "1");
    constexpr cudf::size_type num_rows_per_chunk = 100'000'000;
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);
    {
        auto comm = std::make_shared<rapidsmpf::MPI>(mpi_comm, options);
        auto progress =
            std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);
        auto ctx =
            std::make_shared<rapidsmpf::streaming::Context>(options, comm, &br, stats);
        comm->logger().print(
            "Executor has ", ctx->executor()->thread_count(), " threads"
        );
        comm->logger().print("Executor has ", ctx->comm()->nranks(), " ranks");

        bool is_dir = argc > 1;
        bool is_unordered = argc > 2;
        for (int i = 0; i < 2; i++) {
            std::vector<rapidsmpf::streaming::Node> nodes;
            auto start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing read_lineitem pipeline");
                auto lineitem = ctx->create_channel();
                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    /* num_producers */ 4,
                    num_rows_per_chunk,
                    is_dir,
                    is_unordered
                ));
                nodes.push_back(consume(ctx, lineitem));
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> pipeline = end - start;
            start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("read_lineitem Iteration");
                rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
            }
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double> compute = end - start;
            comm->logger().print(
                "Iteration ", i, " pipeline construction time [s]: ", pipeline.count()
            );
            comm->logger().print("Iteration ", i, " compute time [s]: ", compute.count());
            ctx->comm()->logger().print(stats->report());
            RAPIDSMPF_MPI(MPI_Barrier(mpi_comm));
        }
    }
    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
