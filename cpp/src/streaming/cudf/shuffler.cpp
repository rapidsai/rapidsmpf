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
 *
 * TODO: move to a global place so it can be used throughout RapidsMPF.
 */
class CudaEventRAII {
  public:
    CudaEventRAII(unsigned flags = cudaEventDisableTiming) {
        RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEventRAII() noexcept {
        if (event_) {
            (void)cudaEventDestroy(event_);
        }  // don't throw from dtor
    }

    CudaEventRAII(CudaEventRAII const&) = delete;
    CudaEventRAII& operator=(CudaEventRAII const&) = delete;
    CudaEventRAII(CudaEventRAII&&) = delete;
    CudaEventRAII& operator=(CudaEventRAII&&) = delete;

    /**
     * @brief Get the wrapped event.
     *
     * @return The underlying event.
     */
    [[nodiscard]] cudaEvent_t const& value() const noexcept {
        return event_;
    }

    /**
     * @brief Implicit conversion to a cudaEvent_t reference.
     *
     * @return Const reference to the underlying event.
     */
    operator cudaEvent_t const&() const noexcept {
        return event_;
    }

  private:
    cudaEvent_t event_{};
};

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
        RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(primary, event, 0));
    }
}

}  // namespace

Node shuffler(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    SharedChannel<PartitionMapChunk> ch_in,
    SharedChannel<PartitionVectorChunk> ch_out,
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
    CudaEventRAII event;

    std::uint64_t sequence_number{0};
    while (true) {
        auto partition_map = co_await ch_in->receive_or({});
        if (partition_map == nullptr) {
            break;
        }
        // Make sure that the input chunk's stream is in sync with shuffler's stream.
        sync_streams(stream, partition_map->stream, event);

        shuffler.insert(std::move(partition_map->data));

        // Use the highest input sequence number as the output sequence number.
        sequence_number = std::max(sequence_number, partition_map->sequence_number);
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
                sequence_number, std::move(packed_chunks)
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}


}  // namespace rapidsmpf::streaming::node
