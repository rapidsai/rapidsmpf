/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <optional>
#include <string>

#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

/**
 * @brief Stream-ordered wall-clock timer that records its result into Statistics.
 *
 * Marks a start position in the CUDA stream on construction and a stop position
 * when `stop_and_record` is called.  The elapsed wall-clock time between those two
 * stream positions is recorded into the supplied `Statistics` object once the stream
 * reaches the stop marker — guaranteeing that the measurement covers exactly the
 * work enqueued between the two calls, in stream order.
 *
 * If `statistics` is `Statistics::disabled()`, the entire class is a no-op.
 *
 * @code
 * StreamOrderedTiming timing{stream, stats};
 * // ... enqueue GPU work on `stream` ...
 * timing.stop_and_record("my-operation-time");
 * @endcode
 */
class StreamOrderedTiming {
  public:
    /**
     * @brief Constructs a StreamOrderedTiming and marks the start position in the stream.
     *
     * If `statistics` is `Statistics::disabled()`, this is a no-op and subsequent
     * calls to `stop_and_record` will also be no-ops.
     *
     * @param stream The CUDA stream to time.
     * @param statistics The Statistics object that will receive the duration entry.
     */
    StreamOrderedTiming(
        rmm::cuda_stream_view stream, std::shared_ptr<Statistics> statistics
    );

    /**
     * @brief Marks the stop position in the stream and schedules recording of the
     * duration.
     *
     * The stream-ordered duration (time between the start and stop stream positions) is
     * recorded under @p name. If @p stream_delay_name is set, the **stream delay**
     * — the wall-clock time between object construction and when the stream actually
     * executed the start callback — is also recorded under that name. The stream delay
     * reveals how far ahead the CPU is running relative to the GPU stream.
     *
     * Both values are written to the Statistics object in stream order — i.e. only after
     * all work enqueued between construction and this call has been reached by the
     * stream. If the Statistics object is destroyed before that point, the recording is
     * silently skipped.
     *
     * Behaviour is undefined if this method is called more than once per
     * `StreamOrderedTiming` instance.
     *
     * @param name Name of the stream-ordered duration statistic.
     * @param stream_delay_name Name of the stream-delay statistic. If `std::nullopt`
     * (the default), no stream-delay entry is written.
     */
    void stop_and_record(
        std::string const& name,
        std::optional<std::string> stream_delay_name = std::nullopt
    );

    /**
     * @brief Cancel all in-flight timings associated with a Statistics object.
     *
     * Should be called when a Statistics object is about to be destroyed, to
     * prevent dangling references from any in-flight stream callbacks. It is safe
     * to call when no in-flight timings are present.
     *
     * @note If a stop callback has already executed before this function is called,
     * the associated statistic may still be recorded. The guarantee is only that any
     * still-pending callbacks are cancelled.
     *
     * @param statistics The Statistics object whose in-flight timings should be
     * cancelled.
     */
    static void cancel_inflight_timings(Statistics const* statistics);

  private:
    std::uintptr_t uid_{0};
    rmm::cuda_stream_view stream_;
    std::shared_ptr<Statistics> statistics_;
};

}  // namespace rapidsmpf
