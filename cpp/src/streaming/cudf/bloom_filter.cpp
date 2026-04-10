/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <cudf/stream_compaction.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/bloom_filter.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allreduce.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/cudf/bloom_filter.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {
Actor BloomFilter::build(
    std::shared_ptr<Channel> ch_in, std::shared_ptr<Channel> ch_out, OpID tag
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx_->executor()->schedule();
    co_await ch_in->shutdown_metadata();
    co_await ch_out->shutdown_metadata();
    auto const& br = ctx_->br();
    auto mr = br->device_mr();
    auto filter_stream = br->stream_pool().get_stream();
    CudaEvent event;
    auto storage = rapidsmpf::BloomFilter::storage(num_filter_blocks_, filter_stream, mr);
    RAPIDSMPF_CUDA_TRY(
        cudaMemsetAsync(storage->data(), 0, storage->size(), filter_stream)
    );
    auto filter =
        rapidsmpf::BloomFilter(num_filter_blocks_, seed_, storage->data(), filter_stream);
    CudaEvent build_event;
    build_event.record(filter_stream);
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = msg.release<TableChunk>();
        chunk = co_await chunk.make_available(
            ctx_, -safe_cast<std::int64_t>(chunk.data_alloc_size(MemoryType::DEVICE))
        );
        // Filter is allocated on `filter_stream`, but we run the additions on the chunk's
        // stream. The addition modifies global memory but we can safely launch two
        // kernels doing that concurrently because the updates are atomic.
        build_event.stream_wait(chunk.stream());
        filter.add(chunk.table_view(), chunk.stream(), mr);
        cuda_stream_join(filter_stream, chunk.stream(), &event);
    }
    if (comm_->nranks() > 1) {
        auto reducer = streaming::AllReduce(
            ctx_,
            comm_,
            br->move(std::move(storage), filter_stream),
            br->move(
                rapidsmpf::BloomFilter::storage(num_filter_blocks_, filter_stream, mr),
                filter_stream
            ),
            tag,
            [num_blocks = num_filter_blocks_,
             seed = seed_](Buffer const* left, Buffer* right) {
                right->write_access([&](std::byte* out_bytes,
                                        rmm::cuda_stream_view stream) {
                    auto const in = rapidsmpf::BloomFilter::view(
                        num_blocks, seed, left->data(), stream
                    );
                    rapidsmpf::BloomFilter(num_blocks, seed, out_bytes, stream)
                        .merge(in, stream);
                });
            }
        );
        auto result = co_await reducer.extract();
        auto [res, _] = br->reserve(MemoryType::DEVICE, 0, AllowOverbooking::YES);
        storage = br->move_to_device_buffer(std::move(result.second), res);
    }
    co_await ch_out->send(Message{0, std::move(storage), {}, {}});
    co_await ch_out->drain(ctx_->executor());
}

Actor BloomFilter::apply(
    std::shared_ptr<Channel> bloom_filter,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::vector<cudf::size_type> keys
) {
    streaming::ShutdownAtExit c{bloom_filter, ch_in, ch_out};
    co_await ctx_->executor()->schedule();
    auto storage = (co_await bloom_filter->receive()).release<rmm::device_buffer>();
    RAPIDSMPF_EXPECTS(
        (co_await bloom_filter->receive()).empty(),
        "Bloom filter channel contained more than one message"
    );
    auto stream = storage.stream();
    CudaEvent event;
    auto filter =
        rapidsmpf::BloomFilter(num_filter_blocks_, seed_, storage.data(), stream);
    auto meta = co_await ch_in->receive_metadata();
    if (!meta.empty()) {
        co_await ch_out->send_metadata(std::move(meta));
    }
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = msg.release<TableChunk>();
        chunk = co_await chunk.make_available(
            ctx_, -safe_cast<std::int64_t>(chunk.data_alloc_size(MemoryType::DEVICE))
        );
        auto chunk_stream = chunk.stream();
        cuda_stream_join(chunk_stream, stream, &event);
        // Reservation for the mask construction and guess at output size.
        auto res = co_await ctx_->memory(MemoryType::DEVICE)
                       ->reserve_or_wait(
                           safe_cast<std::size_t>(chunk.table_view().num_rows())
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
            safe_cast<cudf::size_type>(mask.size()),
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
