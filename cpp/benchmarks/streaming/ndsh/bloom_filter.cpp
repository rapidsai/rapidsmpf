/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "bloom_filter_impl.hpp"
#include "utils.hpp"

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
    auto mr = ctx->br()->device_mr();
    auto stream = ctx->br()->stream_pool().get_stream();
    CudaEvent event;
    auto storage = create_filter_storage(num_blocks, stream, mr);
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(storage.data, 0, storage.size, stream));
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
        ctx->comm()->logger().print((int)chunk.table_view().column(0).type().id());
        cuda_stream_join(chunk.stream(), stream, &event);
        update_filter(storage, num_blocks, chunk.table_view(), seed, chunk.stream(), mr);
        cuda_stream_join(stream, chunk.stream(), &event);
    }

    auto allgather = streaming::AllGather(ctx, tag);
    auto metadata = std::vector<std::uint8_t>(storage.size);
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        metadata.data(), storage.data, storage.size, cudaMemcpyDefault, stream.value()
    ));
    stream.synchronize();
    auto [res, _] = ctx->br()->reserve(MemoryType::HOST, 0, true);
    allgather.insert(
        0,
        {std::make_unique<std::vector<std::uint8_t>>(std::move(metadata)),
         ctx->br()->allocate(stream, std::move(res))}
    );
    allgather.insert_finished();
    auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);
    auto merged = std::make_unique<std::vector<std::byte>>(storage.size);
    for (auto&& data : per_rank) {
        for (std::size_t i = 0; i < storage.size; i++) {
            (*merged)[i] |= static_cast<std::byte>((*data.metadata)[i]);
        }
    }
    co_await ch_out->send(streaming::Message{0, std::move(merged), {}, {}});
    co_await ch_out->drain(ctx->executor());
}

streaming::Node apply_bloom_filter(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> bloom_filter,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys,
    std::uint64_t seed,
    std::size_t num_blocks
) {
    streaming::ShutdownAtExit c{bloom_filter, ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto data = co_await bloom_filter->receive();
    RAPIDSMPF_EXPECTS(!data.empty(), "Bloom filter channel was shutdown");
    auto stream = ctx->br()->stream_pool().get_stream();
    auto storage = create_filter_storage(num_blocks, stream, ctx->br()->device_mr());
    CudaEvent event;
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        storage.data,
        data.get<std::vector<std::byte>>().data(),
        storage.size,
        cudaMemcpyDefault,
        stream
    ));
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
        auto chunk_stream = chunk.stream();
        cuda_stream_join(chunk_stream, stream, &event);
        auto mask = apply_filter(
            storage,
            num_blocks,
            chunk.table_view().select(keys),
            seed,
            chunk_stream,
            ctx->br()->device_mr()
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
        ctx->comm()->logger().print(
            "Sending filtered chunk ",
            result->num_rows(),
            " before ",
            chunk.table_view().num_rows()
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
