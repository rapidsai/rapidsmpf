/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>

#include <cudf/partitioning.hpp>

#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {


Node partition_and_pack(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::vector<cudf::size_type> columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed
) {
    ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto table = msg.release<TableChunk>();
        auto reservation = ctx->br()->reserve_device_memory_and_spill(
            table.make_available_cost(), false
        );
        auto tbl = table.make_available(reservation);

        PartitionMapChunk partition_map{
            .data = rapidsmpf::partition_and_pack(
                tbl.table_view(),
                std::move(columns_to_hash),
                num_partitions,
                hash_function,
                seed,
                tbl.stream(),
                ctx->br(),
                ctx->statistics()
            )
        };

        co_await ch_out->send(to_message(
            msg.sequence_number(),
            std::make_unique<PartitionMapChunk>(std::move(partition_map))
        ));
    }
    co_await ch_out->drain(ctx->executor());
}

Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }

        // If receiving a partition map, we convert it to a vector and discard
        // partition IDs.
        std::uint64_t seq = msg.sequence_number();
        std::vector<PackedData> data;
        if (msg.holds<PartitionMapChunk>()) {
            auto partition_map = msg.release<PartitionMapChunk>();
            data = to_vector(std::move(partition_map.data));
        } else {
            auto partition_vec = msg.release<PartitionVectorChunk>();
            data = std::move(partition_vec.data);
        }
        // Get a stream for the concatenated table chunk.
        auto stream = ctx->br()->stream_pool().get_stream();

        std::unique_ptr<cudf::table> ret = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(std::move(data), ctx->br(), false),
            stream,
            ctx->br(),
            ctx->statistics()
        );
        co_await ch_out->send(
            to_message(seq, std::make_unique<TableChunk>(std::move(ret), stream))
        );
    }
    co_await ch_out->drain(ctx->executor());
}

}  // namespace rapidsmpf::streaming::node
