/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <memory>
#include <numeric>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>

namespace rapidsmpf::streaming::node {

namespace {

/**
 * @brief Make @p primary wait until all work currently enqueued on @p secondary
 * completes.
 *
 * Records @p event on @p secondary and inserts a wait for that event on @p primary.
 * This is fully asynchronous with respect to the host thread; no host-side blocking.
 *
 * @param primary The stream that must not run ahead.
 * @param secondary The stream whose already-enqueued work must complete first.
 * @param event The CUDA event to use for synchronization. The same event may be reused
 * across multiple calls; the caller does not need to provide an unique event each time.
 */
void sync_streams(
    rmm::cuda_stream_view primary,
    rmm::cuda_stream_view secondary,
    cudaEvent_t const& event
) {
    if (primary.value() != secondary.value()) {
        RAPIDSMPF_CUDA_TRY(cudaEventRecord(event, secondary));
        RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(primary, event));
    }
}

}  // namespace

Node shuffler(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    rapidsmpf::shuffler::Shuffler shuffler(
        ctx->comm(),
        ctx->progress_thread(),
        op_id,
        total_num_partitions,
        stream,
        ctx->br(),
        ctx->statistics(),
        partition_owner
    );
    CudaEvent event;

    std::uint64_t sequence_number{0};
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto partition_map = msg.release<PartitionMapChunk>();

        // Make sure that the input chunk's stream is in sync with shuffler's stream.
        sync_streams(stream, partition_map.stream, event);

        shuffler.insert(std::move(partition_map.data));

        // Use the highest input sequence number as the output sequence number.
        sequence_number = std::max(sequence_number, partition_map.sequence_number);
    }

    // Tell the shuffler that we have no more input data.
    std::vector<rapidsmpf::shuffler::PartID> finished(total_num_partitions);
    std::iota(finished.begin(), finished.end(), 0);
    shuffler.insert_finished(std::move(finished));

    while (!shuffler.finished()) {
        auto finished_partition = shuffler.wait_any();
        auto packed_chunks = shuffler.extract(finished_partition);
        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(
                sequence_number, std::move(packed_chunks), stream
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

std::pair<Node, Node> shuffler_nb(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    // make a shared_ptr to the shuffler so that it can be passed into multiple coroutines
    auto shuffler = std::make_shared<rapidsmpf::shuffler::Shuffler>(
        ctx->comm(),
        ctx->progress_thread(),
        op_id,
        total_num_partitions,
        stream,
        ctx->br(),
        ctx->statistics(),
        std::move(partition_owner)
    );

    // insert task: insert the partition map chunks into the shuffler
    auto insert_task =
        [](
            auto shuffler, auto ctx, auto total_num_partitions, auto stream, auto ch_in
        ) -> Node {
        ShutdownAtExit c{ch_in};
        co_await ctx->executor()->schedule();
        CudaEvent event;

        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            auto partition_map = msg.template release<PartitionMapChunk>();

            // Make sure that the input chunk's stream is in sync with shuffler's stream.
            sync_streams(stream, partition_map.stream, event);

            shuffler->insert(std::move(partition_map.data));
        }

        // Tell the shuffler that we have no more input data.
        std::vector<rapidsmpf::shuffler::PartID> finished(total_num_partitions);
        std::iota(finished.begin(), finished.end(), 0);
        shuffler->insert_finished(std::move(finished));
        co_return;
    };

    // extract task: extract the packed chunks from the shuffler and send them to the
    // output channel
    auto extract_task = [](auto shuffler, auto ctx, auto ch_out) -> Node {
        ShutdownAtExit c{ch_out};
        co_await ctx->executor()->schedule();

        coro::mutex mtx{};
        coro::condition_variable cv{};
        bool finished{false};

        shuffler->register_finished_callback(
            [shuffler, ctx, ch_out, &mtx, &cv, &finished](auto pid) {
                // task to extract and send each finished partition
                auto extract_and_send = [](auto shuffler,
                                           auto ctx,
                                           auto ch_out,
                                           auto pid,
                                           coro::condition_variable& cv,
                                           coro::mutex& mtx,
                                           bool& finished) -> Node {
                    co_await ctx->executor()->schedule();
                    auto packed_chunks = shuffler->extract(pid);
                    co_await ch_out->send(
                        std::make_unique<PartitionVectorChunk>(
                            pid, std::move(packed_chunks)
                        )
                    );

                    // signal that all partitions have been finished
                    if (shuffler->finished()) {
                        {
                            auto lock = co_await mtx.scoped_lock();
                            finished = true;
                        }
                        co_await cv.notify_one();
                    }
                };
                // schedule a detached task to extract and send the packed chunks
                ctx->executor()->spawn(
                    extract_and_send(shuffler, ctx, ch_out, pid, cv, mtx, finished)
                );
            }
        );

        // wait for all partitions to be finished
        {
            auto lock = co_await mtx.scoped_lock();
            co_await cv.wait(lock, [&finished]() { return finished; });
        }

        co_await ch_out->drain(ctx->executor());
    };

    return {
        insert_task(shuffler, ctx, total_num_partitions, stream, std::move(ch_in)),
        extract_task(std::move(shuffler), std::move(ctx), std::move(ch_out))
    };
}

}  // namespace rapidsmpf::streaming::node
