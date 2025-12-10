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
    co_await ctx->executor()->schedule();
    auto mr = ctx->br()->device_mr();
    auto stream = ctx->br()->stream_pool().get_stream();
    CudaEvent event;
    auto storage = create_filter_storage(num_blocks, stream, mr);
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(storage.data, 0, storage.size, stream));
    CudaEvent storage_event;
    storage_event.record(stream);
    auto start = Clock::now();
    bool started = false;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (!started) {
            start = Clock::now();
            started = true;
        }
        if (msg.empty()) {
            break;
        }
        auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
        storage_event.stream_wait(chunk.stream());
        update_filter(storage, num_blocks, chunk.table_view(), seed, chunk.stream(), mr);
        cuda_stream_join(stream, chunk.stream(), &event);
    }

    ctx->comm()->logger().print(
        "Bloom filter of ", storage.size, " bytes local build took ", Clock::now() - start
    );
    auto t0 = Clock::now();
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(1);
    auto [res, _] = ctx->br()->reserve(MemoryType::DEVICE, storage.size, true);
    auto buf = ctx->br()->allocate(stream, std::move(res));
    buf->write_access([&](std::byte* data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            data, storage.data, storage.size, cudaMemcpyDefault, stream.value()
        ));
    });
    ctx->comm()->logger().print(
        "Bloom filter allocate and copy to buf took ", t0 - start
    );
    t0 = Clock::now();
    auto allgather = streaming::AllGather(ctx, tag);
    allgather.insert(0, {std::move(metadata), std::move(buf)});
    ctx->comm()->logger().print(
        "Bloom filter allgather insertion ", Clock::now() - t0, " ", Clock::now() - start
    );
    t0 = Clock::now();
    allgather.insert_finished();
    auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);
    ctx->comm()->logger().print(
        "Bloom filter extract all took ", Clock::now() - t0, " ", Clock::now() - start
    );
    auto temp_storage = create_filter_storage(num_blocks, stream, mr);
    for (auto&& data : per_rank) {
        cuda_stream_join(data.data->stream(), stream, &event);
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            temp_storage.data,
            data.data->data(),
            temp_storage.size,
            cudaMemcpyDefault,
            stream
        ));
        cuda_stream_join(stream, data.data->stream(), &event);
        merge_filters(storage, temp_storage, num_blocks, stream);
    }
    ctx->comm()->logger().print("Bloom filter build took ", Clock::now() - start);
    co_await ch_out->send(
        streaming::Message{
            0,
            std::make_unique<rapidsmpf::ndsh::aligned_buffer>(std::move(storage)),
            {},
            {}
        }
    );
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
    auto storage = data.release<rapidsmpf::ndsh::aligned_buffer>();
    auto stream = storage.stream;
    CudaEvent event;
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
