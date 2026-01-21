/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
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

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "utils.hpp"

namespace rapidsmpf::ndsh {

coro::task<streaming::Message> broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    OpID tag,
    streaming::AllGather::Ordered ordered
) {
    streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    CudaEvent event;
    ctx->comm()->logger().print("Broadcast ", static_cast<int>(tag));
    if (ctx->comm()->nranks() == 1) {
        std::vector<streaming::TableChunk> chunks;
        std::vector<cudf::table_view> views;
        auto gather_stream = ctx->br()->stream_pool().get_stream();
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            auto chunk = to_device(ctx, msg.release<streaming::TableChunk>());
            cuda_stream_join(gather_stream, chunk.stream(), &event);
            views.push_back(chunk.table_view());
            chunks.push_back(std::move(chunk));
        }
        if (chunks.size() == 1) {
            co_return streaming::to_message(
                0, std::make_unique<streaming::TableChunk>(std::move(chunks[0]))
            );
        } else {
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
                break;
            }
            // TODO: If this chunk is already in pack form, this is unnecessary.
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
        auto result = co_await gatherer.extract_all(ordered);
        if (result.size() == 1) {
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    std::make_unique<PackedData>(std::move(result[0]))
                )
            );
        } else {
            auto stream = ctx->br()->stream_pool().get_stream();
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    unpack_and_concat(
                        unspill_partitions(
                            std::move(result),
                            ctx->br().get(),
                            AllowOverbooking::YES,
                            ctx->statistics()
                        ),
                        stream,
                        ctx->br().get(),
                        ctx->statistics()
                    ),
                    stream
                )
            );
        }
    }
}

streaming::Node broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    OpID tag,
    streaming::AllGather::Ordered ordered
) {
    streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    co_await ch_out->send(co_await broadcast(ctx, ch_in, tag, ordered));
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Join a table chunk against a build hash table returning a message of the result.
 *
 * @param ctx Streaming context
 * @param right_chunk Chunk to join
 * @param sequence Sequence number of the output
 * @param joiner hash_join object, representing the build table.
 * @param build_carrier Columns from the build-side table to be included in the output.
 * @param right_on Key column indiecs in `right_chunk`.
 * @param build_stream Stream the `joiner` will be deallocated on.
 * @param build_event Event recording the creation of the `joiner`.
 *
 * @return Message of `TableChunk` containing the result of the inner join.
 */
streaming::Message inner_join_chunk(
    std::shared_ptr<streaming::Context> ctx,
    streaming::TableChunk&& right_chunk,
    std::uint64_t sequence,
    cudf::hash_join& joiner,
    cudf::table_view build_carrier,
    std::vector<cudf::size_type> right_on,
    rmm::cuda_stream_view build_stream,
    CudaEvent* build_event
) {
    CudaEvent event;
    right_chunk = to_device(ctx, std::move(right_chunk));
    auto chunk_stream = right_chunk.stream();
    build_event->stream_wait(chunk_stream);
    auto probe_table = right_chunk.table_view();
    auto probe_keys = probe_table.select(right_on);
    auto [probe_match, build_match] =
        joiner.inner_join(probe_keys, std::nullopt, chunk_stream, ctx->br()->device_mr());

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
    ctx->comm()->logger().print("Inner broadcast join ", static_cast<int>(tag));
    auto build_table = to_device(
        ctx,
        (co_await broadcast(ctx, left, tag, streaming::AllGather::Ordered::NO))
            .release<streaming::TableChunk>()
    );
    ctx->comm()->logger().print(
        "Build table has ", build_table.table_view().num_rows(), " rows"
    );

    auto joiner = cudf::hash_join(
        build_table.table_view().select(left_on),
        cudf::null_equality::UNEQUAL,
        build_table.stream()
    );
    CudaEvent build_event;
    build_event.record(build_table.stream());
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
    while (!ch_out->is_shutdown()) {
        auto right_msg = co_await right->receive();
        if (right_msg.empty()) {
            break;
        }
        co_await ch_out->send(inner_join_chunk(
            ctx,
            right_msg.release<streaming::TableChunk>(),
            right_msg.sequence_number(),
            joiner,
            build_carrier,
            right_on,
            build_table.stream(),
            &build_event
        ));
    }

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
    ctx->comm()->logger().print("Inner shuffle join");
    co_await ctx->executor()->schedule();
    CudaEvent build_event;
    while (!ch_out->is_shutdown()) {
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
        build_event.record(build_stream);
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
            build_stream,
            &build_event
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
    ctx->comm()->logger().print("Shuffle ", static_cast<int>(tag));
    streaming::ShufflerAsync shuffler(ctx, tag, num_partitions);
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
            ctx->br().get(),
            ctx->statistics()
        );
        shuffler.insert(std::move(packed));
    }
    co_await shuffler.insert_finished();
    for (auto pid : shuffler.local_partitions()) {
        auto packed_data = co_await shuffler.extract_async(pid);
        RAPIDSMPF_EXPECTS(packed_data.has_value(), "Partition already extracted");
        auto stream = ctx->br()->stream_pool().get_stream();
        co_await ch_out->send(
            streaming::to_message(
                pid,
                std::make_unique<streaming::TableChunk>(
                    unpack_and_concat(
                        unspill_partitions(
                            std::move(*packed_data),
                            ctx->br().get(),
                            AllowOverbooking::YES,
                            ctx->statistics()
                        ),
                        stream,
                        ctx->br().get(),
                        ctx->statistics()
                    ),
                    stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

namespace {
coro::task<std::pair<std::size_t, std::size_t>> allgather_join_sizes(
    std::shared_ptr<streaming::Context> ctx,
    OpID tag,
    std::size_t left_local_bytes,
    std::size_t right_local_bytes
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(2 * sizeof(std::size_t));
    std::memcpy(metadata->data(), &left_local_bytes, sizeof(left_local_bytes));
    std::memcpy(
        metadata->data() + sizeof(left_local_bytes),
        &right_local_bytes,
        sizeof(right_local_bytes)
    );

    auto stream = ctx->br()->stream_pool().get_stream();
    auto [res, _] = ctx->br()->reserve(MemoryType::HOST, 0, true);
    auto buf = ctx->br()->allocate(stream, std::move(res));
    auto allgather = streaming::AllGather(ctx, tag);
    allgather.insert(0, {std::move(metadata), std::move(buf)});
    allgather.insert_finished();
    auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);

    std::size_t left_total_bytes = 0;
    std::size_t right_total_bytes = 0;
    for (auto const& data : per_rank) {
        RAPIDSMPF_EXPECTS(
            data.metadata->size() >= 2 * sizeof(std::size_t),
            "Invalid metadata size for adaptive join size estimation"
        );
        std::size_t bytes = 0;
        std::memcpy(&bytes, data.metadata->data(), sizeof(bytes));
        left_total_bytes += bytes;
        std::memcpy(&bytes, data.metadata->data() + sizeof(bytes), sizeof(bytes));
        right_total_bytes += bytes;
    }
    co_return {left_total_bytes, right_total_bytes};
}

streaming::Node replay_channel(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> input,
    std::shared_ptr<streaming::Channel> output,
    std::vector<streaming::Message> buffer
) {
    streaming::ShutdownAtExit c{input, output};
    co_await ctx->executor()->schedule();
    for (auto&& msg : buffer) {
        if (msg.empty()) {
            co_await output->drain(ctx->executor());
            co_return;
        }
        if (!co_await output->send(std::move(msg))) {
            co_return;
        }
    }
    while (!output->is_shutdown()) {
        auto msg = co_await input->receive();
        if (msg.empty()) {
            break;
        }
        co_await output->send(std::move(msg));
    }
    co_await output->drain(ctx->executor());
}
}  // namespace

streaming::Node adaptive_inner_join(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> left,
    std::shared_ptr<streaming::Channel> right,
    std::shared_ptr<streaming::Channel> left_meta,
    std::shared_ptr<streaming::Channel> right_meta,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_keys,
    std::vector<cudf::size_type> right_keys,
    OpID allreduce_tag,
    OpID left_shuffle_tag,
    OpID right_shuffle_tag
) {
    streaming::ShutdownAtExit c{left, right, left_meta, right_meta, ch_out};
    co_await ctx->executor()->schedule();

    auto consume_meta = [&](
                            std::shared_ptr<streaming::Channel> ch
                        ) -> coro::task<std::optional<std::size_t>> {
        auto msg = co_await ch->receive();
        if (msg.empty()) {
            co_return std::nullopt;
        }
        co_return msg.release<std::size_t>();
    };
    auto [num_left_messages, num_right_messages] = streaming::coro_results(
        co_await coro::when_all(consume_meta(left_meta), consume_meta(right_meta))
    );
    // Assumption: the input channels carry only TableChunk messages.
    // Assumption: summing data_alloc_size across memory types is a good proxy for the
    // amount of data that will need to be materialized on device for compute.
    // Assumption: a small sample of chunks is representative of the whole table size.
    // Assumption: metadata estimates reflect the total number of chunks per input.
    constexpr std::size_t broadcast_cap_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr double broadcast_ratio_threshold = 0.10;
    constexpr std::size_t inspect_messages = 2;
    std::vector<streaming::Message> left_buffer;
    std::vector<streaming::Message> right_buffer;
    left_buffer.reserve(inspect_messages);
    right_buffer.reserve(inspect_messages);

    auto inspect_channel =
        [&](std::shared_ptr<streaming::Channel> ch,
            std::vector<streaming::Message>& buffer,
            std::size_t estimated_num_messages) -> coro::task<std::size_t> {
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < inspect_messages; ++i) {
            auto msg = co_await ch->receive();
            if (msg.empty()) {
                buffer.push_back(std::move(msg));
                co_return bytes;
            }
            auto const& chunk = msg.get<streaming::TableChunk>();
            for (auto mem_type : MEMORY_TYPES) {
                bytes += chunk.data_alloc_size(mem_type);
            }
            buffer.push_back(std::move(msg));
        }
        co_return (bytes * estimated_num_messages) / inspect_messages;
    };

    auto left_local_bytes = co_await inspect_channel(
        left, left_buffer, num_left_messages.value_or(inspect_messages)
    );
    auto right_local_bytes = co_await inspect_channel(
        right, right_buffer, num_right_messages.value_or(inspect_messages)
    );

    std::size_t left_total_bytes = left_local_bytes;
    std::size_t right_total_bytes = right_local_bytes;
    if (ctx->comm()->nranks() > 1) {
        std::tie(left_total_bytes, right_total_bytes) = co_await allgather_join_sizes(
            ctx, allreduce_tag, left_local_bytes, right_local_bytes
        );
    }

    ctx->comm()->logger().print(
        "Adaptive join total sizes: left ",
        left_total_bytes,
        " bytes, right ",
        right_total_bytes,
        " bytes"
    );
    auto const min_bytes = std::min(left_total_bytes, right_total_bytes);
    auto const max_bytes = std::max(left_total_bytes, right_total_bytes);
    auto const broadcast_ratio =
        (max_bytes == 0) ? 0.0 : static_cast<double>(min_bytes) / max_bytes;
    auto const use_broadcast =
        min_bytes <= broadcast_cap_bytes && broadcast_ratio <= broadcast_ratio_threshold;
    auto left_replay = ctx->create_channel();
    auto right_replay = ctx->create_channel();
    std::vector<streaming::Node> tasks;
    tasks.push_back(replay_channel(ctx, left, left_replay, std::move(left_buffer)));
    tasks.push_back(replay_channel(ctx, right, right_replay, std::move(right_buffer)));
    if (use_broadcast) {
        ctx->comm()->logger().print("Adaptive join strategy: broadcast");
        auto const broadcast_left = left_total_bytes <= right_total_bytes;
        auto build = broadcast_left ? left_replay : right_replay;
        auto probe = broadcast_left ? right_replay : left_replay;
        auto build_keys = broadcast_left ? left_keys : right_keys;
        auto probe_keys = broadcast_left ? right_keys : left_keys;
        auto const broadcast_tag = broadcast_left ? left_shuffle_tag : right_shuffle_tag;
        tasks.push_back(inner_join_broadcast(
            ctx,
            build,
            probe,
            ch_out,
            std::move(build_keys),
            std::move(probe_keys),
            broadcast_tag
        ));
    } else {
        ctx->comm()->logger().print("Adaptive join strategy: shuffle");
        auto const num_partitions = static_cast<std::uint32_t>(ctx->comm()->nranks());
        auto left_shuffled = ctx->create_channel();
        auto right_shuffled = ctx->create_channel();
        tasks.push_back(shuffle(
            ctx, left_replay, left_shuffled, left_keys, num_partitions, left_shuffle_tag
        ));
        tasks.push_back(shuffle(
            ctx,
            right_replay,
            right_shuffled,
            right_keys,
            num_partitions,
            right_shuffle_tag
        ));
        tasks.push_back(inner_join_shuffle(
            ctx, left_shuffled, right_shuffled, ch_out, left_keys, right_keys
        ));
    }
    streaming::coro_results(co_await coro::when_all(std::move(tasks)));
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::ndsh
