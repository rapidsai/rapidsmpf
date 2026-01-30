/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bloom_filter.hpp"

#include <memory>
#include <vector>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::ndsh {
streaming::Node build_bloom_filter(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    OpID tag,
    std::uint64_t seed,
    std::size_t num_blocks
) {
    streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto mr = ctx->br()->device_mr();
    auto stream = ctx->br()->stream_pool().get_stream();
    CudaEvent event;
    auto filter = std::make_unique<BloomFilter>(num_blocks, seed, stream, mr);
    CudaEvent build_event;
    build_event.record(stream);
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = co_await msg.release<streaming::TableChunk>().make_available(ctx);
        build_event.stream_wait(chunk.stream());
        filter->add(chunk.table_view(), chunk.stream(), mr);
        cuda_stream_join(stream, chunk.stream(), &event);
    }

    if (ctx->comm()->nranks() > 1) {
        auto metadata = std::make_unique<std::vector<std::uint8_t>>(1);
        auto [res, _] =
            ctx->br()->reserve(MemoryType::DEVICE, filter->size(), AllowOverbooking::YES);
        auto buf = ctx->br()->allocate(stream, std::move(res));
        buf->write_access([&](std::byte* data, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                data, filter->data(), filter->size(), cudaMemcpyDefault, stream.value()
            ));
        });
        auto allgather = streaming::AllGather(ctx, tag);
        allgather.insert(0, {std::move(metadata), std::move(buf)});
        allgather.insert_finished();
        auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);
        auto other = BloomFilter(num_blocks, seed, stream, mr);
        for (auto&& data : per_rank) {
            cuda_stream_join(data.data->stream(), stream, &event);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                other.data(), data.data->data(), other.size(), cudaMemcpyDefault, stream
            ));
            cuda_stream_join(stream, data.data->stream(), &event);
            filter->merge(other, stream);
        }
    }
    co_await ch_out->send(streaming::Message{0, std::move(filter), {}, {}});
    co_await ch_out->drain(ctx->executor());
}

streaming::Node apply_bloom_filter(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> bloom_filter,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys
) {
    streaming::ShutdownAtExit c{bloom_filter, ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto data = co_await bloom_filter->receive();
    RAPIDSMPF_EXPECTS(!data.empty(), "Bloom filter channel was shutdown");
    auto filter = data.release<BloomFilter>();
    auto stream = filter.stream();
    CudaEvent event;
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = co_await msg.release<streaming::TableChunk>().make_available(ctx);
        auto chunk_stream = chunk.stream();
        cuda_stream_join(chunk_stream, stream, &event);
        auto mask = filter.contains(
            chunk.table_view().select(keys), chunk_stream, ctx->br()->device_mr()
        );
        cuda_stream_join(stream, chunk_stream, &event);
        RAPIDSMPF_EXPECTS(
            mask.size() == static_cast<std::size_t>(chunk.table_view().num_rows()),
            "Invalid mask size"
        );
        auto mask_view = cudf::column_view{
            cudf::data_type{cudf::type_id::BOOL8},
            static_cast<cudf::size_type>(mask.size()),
            mask.data(),
            {},
            0
        };
        auto result = cudf::apply_boolean_mask(
            chunk.table_view(), mask_view, chunk_stream, ctx->br()->device_mr()
        );
        std::ignore = std::move(chunk);
        co_await ch_out->send(to_message(
            msg.sequence_number(),
            std::make_unique<streaming::TableChunk>(std::move(result), chunk_stream)
        ));
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::ndsh
