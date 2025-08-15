/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <memory>
#include <numeric>

#include <rapidsmpf/streaming/cudf/shuffler.hpp>

namespace rapidsmpf::streaming::node {


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

    rapidsmpf::shuffler::Shuffler shuffler(
        ctx->comm(),
        ctx->progress_thread(),
        op_id,
        total_num_partitions,
        ctx->stream(),
        ctx->br(),
        ctx->statistics(),
        partition_owner
    );

    std::uint64_t sequence_number{0};
    while (true) {
        auto partition_map = co_await ch_in->receive_or({});
        if (partition_map == nullptr) {
            break;
        }
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
