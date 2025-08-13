/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <memory>
#include <numeric>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/streaming/cudf/shuffler.hpp>

namespace rapidsmpf::streaming::node {

namespace {

/**
 * @brief RAII wrapper for a CUDA event.
 *
 * Creates a CUDA event on construction and destroys it on destruction.
 */
struct CudaEventRAII {
    cudaEvent_t event{};

    CudaEventRAII(unsigned flags = cudaEventDisableTiming) {
        RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event, flags));
    }

    ~CudaEventRAII() {
        if (event) {
            cudaEventRecord(event);
        }
    }

    CudaEventRAII(const CudaEventRAII&) = delete;
    CudaEventRAII& operator=(const CudaEventRAII&) = delete;
    CudaEventRAII(CudaEventRAII&&) = delete;
    CudaEventRAII& operator=(CudaEventRAII&&) = delete;
};

/**
 * @brief Make @p primary wait until all work currently enqueued on @p secondary
 * completes.
 *
 * Records @p ev on @p secondary and inserts a wait for that event on @p primary.
 * This is fully asynchronous with respect to the host thread—no host-side blocking.
 *
 * @param primary    The stream that must not run ahead.
 * @param secondary  The stream whose already-enqueued work must complete first.
 * @param ev         The CUDA event to use for synchronization.
 *                   The same event may be reused across multiple calls; the caller
 *                   does not need to provide a unique event each time.
 *
 * @note Streams must be on the same device/context. Call this each time you finish
 *       enqueueing a batch on @p secondary and before enqueueing the dependent batch
 *       on @p primary.
 */
void sync_streams(
    rmm::cuda_stream_view primary, rmm::cuda_stream_view secondary, cudaEvent_t ev
) {
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(ev, secondary));
    RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(primary, ev, 0));
}

}  // namespace

Node shuffler(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionMapChunk> ch_in,
    SharedChannel<PartitionVectorChunk> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    // We delay the shuffler and stream creation until we got the first input chunk.
    std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
    rmm::cuda_stream_view stream;
    CudaEventRAII event;

    std::uint64_t sequence_number{0};
    while (true) {
        auto partition_map = co_await ch_in->receive_or({});
        if (partition_map == nullptr) {
            break;
        }

        if (shuffler == nullptr) {
            // The shuffler uses the first chunk's CUDA stream.
            stream = partition_map->stream;
            shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
                ctx->comm(),
                ctx->progress_thread(),
                op_id,
                total_num_partitions,
                stream,
                ctx->br(),
                ctx->statistics(),
                partition_owner
            );
        }
        // If the shuffler's and the input chunk's stream doesn't match, we sync them.
        if (stream.value() != partition_map->stream.value()) {
            sync_streams(stream, partition_map->stream, event.event);
        }

        shuffler->insert(std::move(partition_map->data));

        // Use the highest input sequence number as the output sequence number.
        sequence_number = std::max(sequence_number, partition_map->sequence_number);
    }

    // Tell the shuffler that we have no more input data.
    std::vector<rapidsmpf::shuffler::PartID> finished(total_num_partitions);
    std::iota(finished.begin(), finished.end(), 0);
    shuffler->insert_finished(std::move(finished));

    while (!shuffler->finished()) {
        auto finished_partition = shuffler->wait_any();
        auto packed_chunks = shuffler->extract(finished_partition);
        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(
                sequence_number, std::move(packed_chunks)
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}


}  // namespace rapidsmpf::streaming::node
