/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>

namespace rapidsmpf::streaming::node {


/**
 * @brief Launches a shuffler node for a single shuffle operation.
 *
 * This is a streaming version of `rapidsmpf::shuffler::Shuffler` that operates on
 * packed partition chunks using channels.
 *
 * It consumes partitioned input data from the input channel and redistributes it across
 * partitions based on ownership logic, producing output chunks grouped by partition ID.
 * The shuffle uses GPU memory and communication primitives to operate efficiently
 * in a distributed or multi-threaded environment.
 *
 * @param ctx The streaming context providing communication, memory, stream, and execution
 * resources.
 * @param ch_in Input channel providing packed partition chunks to be shuffled.
 * @param ch_out Output channel where the shuffled result are sent.
 * @param op_id Unique operation ID for this shuffle. Must not be reused until
 *              all nodes have called `Shuffler::shutdown()`.
 * @param total_num_partitions Total number of partitions to shuffle the data into.
 * @param partition_owner Function that maps a partition ID to its owning rank/node.
 *                        Defaults to round-robin ownership.
 *
 * @return A streaming node that completes when the shuffling has finished and the output
 * channel is drained.
 */
Node shuffler(
    std::shared_ptr<Context> ctx,
    SharedChannel<PartitionMapChunk> ch_in,
    SharedChannel<PartitionVectorChunk> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner = shuffler::Shuffler::round_robin
);


}  // namespace rapidsmpf::streaming::node
