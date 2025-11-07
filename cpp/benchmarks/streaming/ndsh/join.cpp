/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "utilities.hpp"

namespace rapidsmpf::ndsh {

namespace {

coro::task<streaming::Message> broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    OpID tag
) {
    streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    CudaEvent event;
    if (ctx->comm()->nranks() == 1) {
        std::vector<streaming::TableChunk> chunks;
        std::vector<cudf::table_view> views;
        auto gather_stream = ctx->br()->stream_pool().get_stream();
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            ctx->comm()->logger().print("Broadcast chunk receive");
            auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
            auto chunk_stream = chunk.stream();
            cuda_stream_join(gather_stream, chunk_stream, &event);
            views.push_back(chunk.table_view());
            chunks.push_back(std::move(chunk));
        }
        if (chunks.size() == 1) {
            co_return streaming::to_message(
                0, std::make_unique<streaming::TableChunk>(std::move(chunks[0]))
            );
        } else {
            ctx->comm()->logger().print("Broadcast concatenation");
            RAPIDSMPF_EXPECTS(chunks.size() > 0, "No chunks in broadcast");
            auto result = cudf::concatenate(views, gather_stream, ctx->br()->device_mr());
            // So that deallocation of the consitutent tables is stream-ordered wrt the
            // concatenation.
            cuda_stream_join(
                chunks
                    | std::views::transform([](auto&& chunk) { return chunk.stream(); }),
                std::ranges::single_view(gather_stream),
                &event
            );
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(std::move(result), gather_stream)
            );
        }
    } else {
        streaming::AllGather gatherer{ctx, tag};
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                ctx->comm()->logger().print(
                    "Inserting finished token for broadcast with tag ",
                    static_cast<int>(tag)
                );
                break;
            }
            ctx->comm()->logger().print("Inserting broadcast chunk");
            auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
            auto pack =
                cudf::pack(chunk.table_view(), chunk.stream(), ctx->br()->device_mr());
            auto packed_data = PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), chunk.stream())
            );
            gatherer.insert(msg.sequence_number(), {std::move(packed_data)});
        }
        gatherer.insert_finished();
        auto result = co_await gatherer.extract_all(streaming::AllGather::Ordered::NO);
        ctx->comm()->logger().print(
            "Finished broadcast allgather with tag, extracting ", static_cast<int>(tag)
        );
        if (result.size() == 1) {
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    std::make_unique<PackedData>(std::move(result[0].data))
                )
            );
        } else {
            auto stream = ctx->br()->stream_pool().get_stream();
            std::vector<PackedData> chunks;
            chunks.reserve(result.size());
            std::ranges::transform(result, std::back_inserter(chunks), [](auto& chunk) {
                return std::move(chunk.data);
            });
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    unpack_and_concat(
                        unspill_partitions(
                            std::move(chunks), ctx->br(), true, ctx->statistics()
                        ),
                        stream,
                        ctx->br(),
                        ctx->statistics()
                    ),
                    stream
                )
            );
        }
    }
}

streaming::Message inner_join_chunk(
    std::shared_ptr<streaming::Context> ctx,
    streaming::TableChunk&& right_chunk,
    std::uint64_t sequence,
    cudf::hash_join& joiner,
    cudf::table_view build_carrier,
    std::vector<cudf::size_type> right_on,
    rmm::cuda_stream_view build_stream
) {
    CudaEvent event;
    right_chunk = to_device(ctx, std::move(right_chunk));
    auto chunk_stream = right_chunk.stream();
    cuda_stream_join(build_stream, chunk_stream, &event);
    auto probe_table = right_chunk.table_view();
    auto probe_keys = probe_table.select(right_on);
    // Must run this join on the build_stream. cuco uses an allocator
    // that is stream-ordered wrt the stream used when constructing
    // the hash_join object.
    auto [probe_match, build_match] =
        joiner.inner_join(probe_keys, std::nullopt, build_stream, ctx->br()->device_mr());
    // We're going to deliver the output chunks on chunk stream, so add a dep.
    cuda_stream_join(chunk_stream, build_stream, &event);

    cudf::column_view build_indices =
        cudf::device_span<cudf::size_type const>(*build_match);
    cudf::column_view probe_indices =
        cudf::device_span<cudf::size_type const>(*probe_match);
    // build_carrier is valid on build_stream, but chunk_stream is
    // waiting for build_stream work to be done, so running this on
    // chunk_stream is fine.
    auto result_columns = cudf::gather(
                              build_carrier,
                              build_indices,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              chunk_stream,
                              ctx->br()->device_mr()
    )
                              ->release();
    // drop key columns from probe table.
    std::vector<cudf::size_type> to_keep;
    std::ranges::copy_if(
        std::ranges::iota_view(0, probe_table.num_columns()),
        std::back_inserter(to_keep),
        [&](auto i) { return std::ranges::find(right_on, i) == right_on.end(); }
    );
    std::ranges::move(
        cudf::gather(
            probe_table.select(to_keep),
            probe_indices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            chunk_stream,
            ctx->br()->device_mr()
        )
            ->release(),
        std::back_inserter(result_columns)
    );
    // Deallocation of the join indices will happen on build_stream, so add stream dep
    // This also ensure deallocation of the hash_join object waits for completion.
    cuda_stream_join(build_stream, chunk_stream, &event);
    return streaming::to_message(
        sequence,
        std::make_unique<streaming::TableChunk>(
            std::make_unique<cudf::table>(std::move(result_columns)), chunk_stream
        )
    );
}

}  // namespace

streaming::Node inner_join_broadcast(
    std::shared_ptr<streaming::Context> ctx,
    // We will always choose left as build table and do "broadcast" joins
    std::shared_ptr<streaming::Channel> left,
    std::shared_ptr<streaming::Channel> right,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on,
    OpID tag,
    KeepKeys keep_keys
) {
    streaming::ShutdownAtExit c{left, right, ch_out};
    co_await ctx->executor()->schedule();
    ctx->comm()->logger().print(
        "Starting broadcast join with tag ", static_cast<int>(tag)
    );
    auto build_table = to_device(
        ctx, (co_await broadcast(ctx, left, tag)).release<streaming::TableChunk>()
    );
    ctx->comm()->logger().print(
        "Build table has ", build_table.table_view().num_rows(), " rows"
    );

    auto joiner = cudf::hash_join(
        build_table.table_view().select(left_on),
        cudf::null_equality::UNEQUAL,
        build_table.stream()
    );
    ctx->comm()->logger().print("Built hash join object for tag ", static_cast<int>(tag));
    cudf::table_view build_carrier;
    if (keep_keys == KeepKeys::YES) {
        build_carrier = build_table.table_view();
    } else {
        std::vector<cudf::size_type> to_keep;
        std::ranges::copy_if(
            std::ranges::iota_view(0, build_table.table_view().num_columns()),
            std::back_inserter(to_keep),
            [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); }
        );
        build_carrier = build_table.table_view().select(to_keep);
    }
    std::size_t sequence = 0;
    co_await ctx->executor()->schedule();
    while (true) {
        auto right_msg = co_await right->receive();
        if (right_msg.empty()) {
            break;
        }
        ctx->comm()->logger().print("Doing chunk join for tag ", static_cast<int>(tag));
        co_await ch_out->send(inner_join_chunk(
            ctx,
            right_msg.release<streaming::TableChunk>(),
            sequence++,
            joiner,
            build_carrier,
            right_on,
            build_table.stream()
        ));
    }

    ctx->comm()->logger().print(
        "Finished broadcast join with tag ", static_cast<int>(tag)
    );

    co_await ch_out->drain(ctx->executor());
}

streaming::Node inner_join_shuffle(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> left,
    std::shared_ptr<streaming::Channel> right,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on,
    KeepKeys keep_keys
) {
    streaming::ShutdownAtExit c{left, right, ch_out};
    co_await ctx->executor()->schedule();
    while (true) {
        // Requirement: two shuffles kick out partitions in the same order
        auto left_msg = co_await left->receive();
        auto right_msg = co_await right->receive();
        if (left_msg.empty()) {
            RAPIDSMPF_EXPECTS(
                right_msg.empty(), "Left does not have same number of partitions as right"
            );
            break;
        }
        RAPIDSMPF_EXPECTS(
            left_msg.sequence_number() == right_msg.sequence_number(),
            "Mismatching sequence numbers"
        );
        // TODO: currently always using left as build table.
        auto build_chunk = to_device(ctx, left_msg.release<streaming::TableChunk>());
        auto build_stream = build_chunk.stream();
        auto joiner = cudf::hash_join(
            build_chunk.table_view().select(left_on),
            cudf::null_equality::UNEQUAL,
            build_stream
        );
        ctx->comm()->logger().print("Built hash join object for chunk shuffle join");
        cudf::table_view build_carrier;
        if (keep_keys == KeepKeys::YES) {
            build_carrier = build_chunk.table_view();
        } else {
            std::vector<cudf::size_type> to_keep;
            std::ranges::copy_if(
                std::ranges::iota_view(0, build_chunk.table_view().num_columns()),
                std::back_inserter(to_keep),
                [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); }
            );
            build_carrier = build_chunk.table_view().select(to_keep);
        }
        co_await ch_out->send(inner_join_chunk(
            ctx,
            right_msg.release<streaming::TableChunk>(),
            left_msg.sequence_number(),
            joiner,
            build_carrier,
            right_on,
            build_stream
        ));
    }
    co_await ch_out->drain(ctx->executor());
}

streaming::Node shuffle(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys,
    std::uint32_t num_partitions,
    OpID tag
) {
    streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    ctx->comm()->logger().print("Starting shuffle with tag ", static_cast<int>(tag));
    streaming::ShufflerAsync shuffler(ctx, tag, num_partitions);
    CudaEvent event;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
        auto packed = partition_and_pack(
            chunk.table_view(),
            keys,
            static_cast<int>(num_partitions),
            cudf::hash_id::HASH_MURMUR3,
            0,
            chunk.stream(),
            ctx->br(),
            ctx->statistics()
        );
        ctx->comm()->logger().print(
            "Inserting chunk into shuffle tag ", static_cast<int>(tag)
        );
        shuffler.insert(std::move(packed));
    }
    std::vector<shuffler::PartID> finished(num_partitions);
    std::iota(finished.begin(), finished.end(), 0);
    co_await shuffler.insert_finished(std::move(finished));
    ctx->comm()->logger().print("Shuffle finished tag ", static_cast<int>(tag));
    for (auto pid : shuffler.local_partitions()) {
        auto packed_data = co_await shuffler.extract_async(pid);
        RAPIDSMPF_EXPECTS(packed_data.has_value(), "Partition already extracted");
        ctx->comm()->logger().print(
            "Extracting chunk shuffle tag ", static_cast<int>(tag)
        );
        auto stream = ctx->br()->stream_pool().get_stream();
        co_await ch_out->send(
            streaming::to_message(
                pid,
                std::make_unique<streaming::TableChunk>(
                    unpack_and_concat(
                        unspill_partitions(
                            std::move(*packed_data), ctx->br(), true, ctx->statistics()
                        ),
                        stream,
                        ctx->br(),
                        ctx->statistics()
                    ),
                    stream
                )
            )
        );
    }
    ctx->comm()->logger().print("Finished shuffle with tag ", static_cast<int>(tag));
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::ndsh
