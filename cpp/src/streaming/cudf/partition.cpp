/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>

#include <cudf/partitioning.hpp>

#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>

namespace rapidsmpf::streaming::node {

Node partition_and_pack(
    std::shared_ptr<Context> ctx,
    SharedChannel<TableChunk> ch_in,
    SharedChannel<PartitionMapChunk> ch_out,
    std::vector<cudf::size_type> columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed
) {
    ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    while (true) {
        auto table = co_await ch_in->receive_or(nullptr);
        if (table == nullptr) {
            break;
        }
        auto reservation = ctx->br()->reserve_and_spill(
            MemoryType::DEVICE, table->make_available_cost(), false
        );
        auto tbl = table->make_available(reservation, table->stream());

        PartitionMapChunk partition_map{
            .sequence_number = tbl.sequence_number(),
            .data = rapidsmpf::partition_and_pack(
                tbl.table_view(),
                std::move(columns_to_hash),
                num_partitions,
                hash_function,
                seed,
                table->stream(),
                ctx->br(),
                ctx->statistics()
            ),
            .stream = tbl.stream()
        };

        co_await ch_out->send(
            std::make_unique<PartitionMapChunk>(std::move(partition_map))
        );
    }
    co_await ch_out->drain(ctx->executor());
}

Node partition_and_pack2(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel2> ch_in,
    std::shared_ptr<Channel2> ch_out,
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
        auto reservation = ctx->br()->reserve_and_spill(
            MemoryType::DEVICE, table->make_available_cost(), false
        );
        auto tbl = table->make_available(reservation, table->stream());

        PartitionMapChunk partition_map{
            .sequence_number = tbl.sequence_number(),
            .data = rapidsmpf::partition_and_pack(
                tbl.table_view(),
                std::move(columns_to_hash),
                num_partitions,
                hash_function,
                seed,
                table->stream(),
                ctx->br(),
                ctx->statistics()
            ),
            .stream = tbl.stream()
        };

        co_await ch_out->send(
            std::make_unique<PartitionMapChunk>(std::move(partition_map))
        );
    }
    co_await ch_out->drain(ctx->executor());
}

Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionMapChunk> ch_in,
    SharedChannel<TableChunk> ch_out
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    while (true) {
        auto partition_map = co_await ch_in->receive_or(nullptr);
        if (partition_map == nullptr) {
            break;
        }
        std::unique_ptr<cudf::table> ret = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(
                to_vector(std::move(partition_map->data)),
                partition_map->stream,
                ctx->br(),
                false
            ),
            partition_map->stream,
            ctx->br(),
            ctx->statistics()
        );
        co_await ch_out->send(
            std::make_unique<TableChunk>(
                partition_map->sequence_number, std::move(ret), partition_map->stream
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

Node unpack_and_concat2(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel2> ch_in,
    std::shared_ptr<Channel2> ch_out
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        std::uint64_t seq;
        std::vector<PackedData> data;
        rmm::cuda_stream_view stream;

        // If receiving a partition map, we convert it to a vector and discards
        // partition IDs.
        if (msg.holds<PartitionMapChunk>()) {
            auto partition_map = msg.release<PartitionMapChunk>();
            seq = partition_map->sequence_number;
            data = to_vector(std::move(partition_map->data));
            stream = partition_map->stream;
        } else {
            auto partition_vec = msg.release<PartitionVectorChunk>();
            seq = partition_vec->sequence_number;
            data = std::move(partition_vec->data);
            stream = partition_vec->stream;
        }
        std::unique_ptr<cudf::table> ret = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(std::move(data), stream, ctx->br(), false),
            stream,
            ctx->br(),
            ctx->statistics()
        );
        co_await ch_out->send(std::make_unique<TableChunk>(seq, std::move(ret), stream));
    }
    co_await ch_out->drain(ctx->executor());
}

Node unpack_and_concat(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionVectorChunk> ch_in,
    SharedChannel<TableChunk> ch_out
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    while (true) {
        auto partition_vec = co_await ch_in->receive_or(nullptr);
        if (partition_vec == nullptr) {
            break;
        }
        std::unique_ptr<cudf::table> ret = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(
                std::move(partition_vec->data), partition_vec->stream, ctx->br(), false
            ),
            partition_vec->stream,
            ctx->br(),
            ctx->statistics()
        );
        co_await ch_out->send(
            std::make_unique<TableChunk>(
                partition_vec->sequence_number, std::move(ret), partition_vec->stream
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

}  // namespace rapidsmpf::streaming::node
