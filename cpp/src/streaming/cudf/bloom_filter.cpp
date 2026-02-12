/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <cudf/stream_compaction.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/bloom_filter.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/cudf/bloom_filter.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {
Node BloomFilter::build(
    std::shared_ptr<Channel> ch_in, std::shared_ptr<Channel> ch_out, OpID tag
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx_->executor()->schedule();
    co_await ch_in->shutdown_metadata();
    co_await ch_out->shutdown_metadata();
    auto mr = ctx_->br()->device_mr();
    auto stream = ctx_->br()->stream_pool().get_stream();
    CudaEvent event;
    auto filter =
        std::make_unique<rapidsmpf::BloomFilter>(num_filter_blocks_, seed_, stream, mr);
    CudaEvent build_event;
    build_event.record(stream);
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = msg.release<TableChunk>();
        chunk = co_await chunk.make_available(
            ctx_, -static_cast<std::int64_t>(chunk.data_alloc_size(MemoryType::DEVICE))
        );
        // Filter is allocated one `stream`, but we run the additions on the chunk's
        // stream. The addition modifies global memory but we can safely launch two
        // kernels doing that concurrently because the updates are atomic.
        build_event.stream_wait(chunk.stream());
        filter->add(chunk.table_view(), chunk.stream(), mr);
        cuda_stream_join(stream, chunk.stream(), &event);
    }
    if (ctx_->comm()->nranks() > 1) {
        auto metadata = std::make_unique<std::vector<std::uint8_t>>(1);
        auto [res, _] = ctx_->br()->reserve(
            MemoryType::DEVICE, filter->size(), AllowOverbooking::YES
        );
        auto buf = ctx_->br()->allocate(stream, std::move(res));
        // TODO: Add ability to take ownership of AlignedBuffer in Buffer.
        buf->write_access([&](std::byte* data, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                data, filter->data(), filter->size(), cudaMemcpyDefault, stream.value()
            ));
        });
        // TODO: Use AllReduce once available. Needs ability to provide output buffer so
        // the alignment is respected.
        auto allgather = streaming::AllGather(ctx_, tag);
        allgather.insert(0, {std::move(metadata), std::move(buf)});
        allgather.insert_finished();
        auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);
        auto other = rapidsmpf::BloomFilter(num_filter_blocks_, seed_, stream, mr);
        for (auto&& data : per_rank) {
            cuda_stream_join(data.data->stream(), stream, &event);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                other.data(), data.data->data(), other.size(), cudaMemcpyDefault, stream
            ));
            cuda_stream_join(stream, data.data->stream(), &event);
            filter->merge(other, stream);
        }
    }
    co_await ch_out->send(Message{0, std::move(filter), {}, {}});
    co_await ch_out->drain(ctx_->executor());
}

Node BloomFilter::apply(
    std::shared_ptr<Channel> bloom_filter,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::vector<cudf::size_type> keys
) {
    streaming::ShutdownAtExit c{bloom_filter, ch_in, ch_out};
    co_await ctx_->executor()->schedule();
    auto filter = (co_await bloom_filter->receive()).release<rapidsmpf::BloomFilter>();
    RAPIDSMPF_EXPECTS(
        (co_await bloom_filter->receive()).empty(),
        "Bloom filter channel contained more than one message"
    );
    auto stream = filter.stream();
    CudaEvent event;
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = msg.release<TableChunk>();
        chunk = co_await chunk.make_available(
            ctx_, -static_cast<std::int64_t>(chunk.data_alloc_size(MemoryType::DEVICE))
        );
        auto chunk_stream = chunk.stream();
        cuda_stream_join(chunk_stream, stream, &event);
        // Reservation for the mask construction and guess at output size.
        auto res = co_await ctx_->memory(MemoryType::DEVICE)
                       ->reserve_or_wait(
                           static_cast<std::size_t>(chunk.table_view().num_rows())
                                   // TODO: no magic numbers: the hashing algorithm in
                                   // `contains` below returns an int64 column.
                                   * (1 + sizeof(std::int64_t))
                               // Guess at how selective the filter is.
                               + chunk.data_alloc_size(MemoryType::DEVICE) / 2,
                           0
                       );
        auto mask = filter.contains(
            chunk.table_view().select(keys), chunk_stream, ctx_->br()->device_mr()
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
            chunk.table_view(), mask_view, chunk_stream, ctx_->br()->device_mr()
        );
        std::ignore = std::move(chunk);
        std::ignore = std::move(res);
        co_await ch_out->send(to_message(
            msg.sequence_number(),
            std::make_unique<streaming::TableChunk>(std::move(result), chunk_stream)
        ));
    }
    co_await ch_out->drain(ctx_->executor());
}
}  // namespace rapidsmpf::streaming
