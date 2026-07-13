/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda/memory_resource>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

BufferResource::BufferResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
    std::optional<PinnedMemoryResource> pinned_mr,
    std::unordered_map<MemoryType, std::int64_t> memory_limits,
    std::optional<Duration> periodic_spill_check,
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool,
    std::shared_ptr<Statistics> statistics
)
    : owning_mr_{std::move(device_mr)},
      pinned_mr_{std::move(pinned_mr)},
      host_mr_{},
      stream_pool_{std::move(stream_pool)},
      spill_manager_{this, periodic_spill_check},
      statistics_{std::move(statistics)} {
    // Default every limit to unlimited, then apply caller overrides.
    for (auto& limit : memory_limits_) {
        limit.store(std::numeric_limits<std::int64_t>::max(), std::memory_order_relaxed);
    }
    for (auto const& [mem_type, limit] : memory_limits) {
        memory_limits_[static_cast<std::size_t>(mem_type)].store(
            limit, std::memory_order_relaxed
        );
    }
    RAPIDSMPF_EXPECTS(stream_pool_ != nullptr, "the stream pool pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
}

std::shared_ptr<BufferResource> BufferResource::create(
    cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
    std::optional<PinnedPoolProperties> pinned_pool_properties,
    std::unordered_map<MemoryType, std::int64_t> memory_limits,
    std::optional<Duration> periodic_spill_check,
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool,
    std::shared_ptr<Statistics> statistics
) {
    std::optional<PinnedMemoryResource> pinned_mr;
    if (pinned_pool_properties.has_value() && is_pinned_memory_resources_supported()) {
        pinned_mr = PinnedMemoryResource{*pinned_pool_properties};
    }

    std::shared_ptr<BufferResource> br{new BufferResource{
        std::move(device_mr),
        std::move(pinned_mr),
        std::move(memory_limits),
        periodic_spill_check,
        std::move(stream_pool),
        std::move(statistics)
    }};

    // Install the back-reference on the owned resources *after* construction so
    // that `weak_from_this()` is valid. Each resource holds only a `weak_ptr`,
    // avoiding a reference cycle. When downstream code promotes a non-owning
    // `resource_ref` returned by `device_mr()`/`host_mr()`/`pinned_mr()` to an
    // owning `cuda::mr::any_resource`, the copy promotes the weak reference to a
    // strong one and keeps this `BufferResource` alive.
    auto const weak = br->weak_from_this();
    br->owning_mr_.set_backref(weak);
    br->host_mr_.set_backref(weak);
    if (br->pinned_mr_.has_value()) {
        br->pinned_mr_->set_backref(weak);
    }
    return br;
}

std::shared_ptr<BufferResource> BufferResource::from_options(
    cuda::mr::any_resource<cuda::mr::device_accessible> mr,
    config::Options options,
    std::shared_ptr<Statistics> statistics
) {
    std::unordered_map<MemoryType, std::int64_t> memory_limits{
        {MemoryType::DEVICE, device_limit_from_options(options)}
    };
    return create(
        std::move(mr),
        pinned_pool_properties_from_options(options),
        std::move(memory_limits),
        periodic_spill_check_from_options(options),
        stream_pool_from_options(options),
        std::move(statistics)
    );
}

std::int64_t BufferResource::memory_available(MemoryType mem_type) const noexcept {
    std::int64_t const limit = memory_limits_[static_cast<std::size_t>(mem_type)].load(
        std::memory_order_acquire
    );
    switch (mem_type) {
    case MemoryType::DEVICE:
        return limit - owning_mr_.current_allocated();
    case MemoryType::PINNED_HOST:
        if (!pinned_mr_.has_value()) {
            return 0;
        } else {
            return limit - pinned_mr_->current_allocated();
        }
    case MemoryType::HOST:
        return limit;
    }
    return std::numeric_limits<std::int64_t>::max();
}

void BufferResource::set_memory_limit(MemoryType mem_type, std::int64_t limit) noexcept {
    memory_limits_[static_cast<std::size_t>(mem_type)].store(
        limit, std::memory_order_release
    );
}

rmm::device_async_resource_ref BufferResource::device_mr() noexcept {
    return rmm::device_async_resource_ref{owning_mr_};
}

RmmResourceAdaptor& BufferResource::device_mr_adaptor() noexcept {
    return owning_mr_;
}

rmm::host_async_resource_ref BufferResource::host_mr() noexcept {
    return host_mr_;
}

rmm::host_device_async_resource_ref BufferResource::pinned_mr() {
    RAPIDSMPF_EXPECTS(
        pinned_mr_, "no pinned memory resource is available", std::invalid_argument
    );
    return *pinned_mr_;
}

std::optional<PinnedMemoryResource> BufferResource::try_pinned_mr() const {
    // Returning by value copies the back-referenced `PinnedMemoryResource`, so the
    // returned handle (and any copy of it) keeps this `BufferResource` alive.
    return pinned_mr_;
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, AllowOverbooking allow_overbooking
) {
    RAPIDSMPF_EXPECTS(
        mem_type != MemoryType::PINNED_HOST || pinned_mr_.has_value(),
        "pinned memory resource is not available",
        std::invalid_argument
    );

    std::int64_t const available = memory_available(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t& reserved = memory_reserved_[static_cast<std::size_t>(mem_type)];

    // Calculate the available memory _after_ the memory has been reserved.
    std::int64_t headroom =
        available - (safe_cast<std::int64_t>(reserved) + safe_cast<std::int64_t>(size));
    // If negative, we are overbooking.
    std::size_t overbooking =
        headroom < 0 ? safe_cast<std::size_t>(std::abs(headroom)) : 0;
    if (overbooking > 0 && allow_overbooking == AllowOverbooking::NO) {
        // Cancel the reservation, overbooking isn't allowed.
        return {MemoryReservation(mem_type, this, 0), overbooking};
    }
    // Make the reservation.
    reserved += size;
    return {MemoryReservation(mem_type, this, size), overbooking};
}

MemoryReservation BufferResource::reserve_device_memory_and_spill(
    std::size_t size, AllowOverbooking allow_overbooking
) {
    // reserve device memory with overbooking
    auto [reservation, ob] = reserve(MemoryType::DEVICE, size, AllowOverbooking::YES);

    // ask the spill manager to make room for overbooking
    if (ob > 0) {
        auto spilled = spill_manager_.spill(ob);
        RAPIDSMPF_EXPECTS(
            allow_overbooking == AllowOverbooking::YES || spilled >= ob,
            "failed to spill enough memory (reserved: " + format_nbytes(size)
                + ", overbooking: " + format_nbytes(ob)
                + ", spilled: " + format_nbytes(spilled) + ")",
            rapidsmpf::reservation_error
        );
    }

    return std::move(reservation);
}

std::size_t BufferResource::release(MemoryReservation& reservation, std::size_t size) {
    std::lock_guard const lock(mutex_);
    RAPIDSMPF_EXPECTS(
        size <= reservation.size_,
        "MemoryReservation(" + format_nbytes(reservation.size_) + ") isn't big enough ("
            + format_nbytes(size) + ")",
        rapidsmpf::reservation_error
    );
    std::size_t& reserved =
        memory_reserved_[static_cast<std::size_t>(reservation.mem_type_)];
    RAPIDSMPF_EXPECTS(reserved >= size, "corrupted reservation stat");
    reserved -= size;
    return reservation.size_ -= size;
}

std::unique_ptr<Buffer> BufferResource::make_buffer(
    std::size_t size, rmm::cuda_stream_view stream, MemoryReservation& reservation
) {
    auto const mem_type = reservation.mem_type_;
    StreamOrderedTiming timing{stream, statistics_};
    std::unique_ptr<Buffer> ret;
    switch (mem_type) {
    case MemoryType::HOST:
        ret = std::unique_ptr<Buffer>(new Buffer(
            std::make_unique<HostBuffer>(size, stream, host_mr()),
            stream,
            MemoryType::HOST
        ));
        break;
    case MemoryType::PINNED_HOST:
        ret = std::unique_ptr<Buffer>(new Buffer(
            std::make_unique<HostBuffer>(size, stream, pinned_mr()),
            stream,
            MemoryType::PINNED_HOST
        ));
        break;
    case MemoryType::DEVICE:
        ret = std::unique_ptr<Buffer>(new Buffer(
            std::make_unique<rmm::device_buffer>(size, stream, device_mr()),
            MemoryType::DEVICE
        ));
        break;
    default:
        RAPIDSMPF_FAIL("MemoryType: unknown");
    }
    release(reservation, size);
    statistics_->record_alloc(mem_type, size, std::move(timing));
    return ret;
}

std::unique_ptr<Buffer> BufferResource::make_buffer(
    rmm::cuda_stream_view stream, MemoryReservation&& reservation
) {
    return make_buffer(reservation.size(), stream, reservation);
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
) {
    auto upstream = data->stream();
    if (upstream.value() != stream.value()) {
        cuda_stream_join(stream, upstream);
        data->set_stream(stream);
    }

    if (is_host_accessible(data->memory_resource())) {
        auto pinned_host_buffer = std::make_unique<HostBuffer>(
            HostBuffer::from_rmm_device_buffer(std::move(data), stream)
        );
        return std::unique_ptr<Buffer>(
            new Buffer(std::move(pinned_host_buffer), stream, MemoryType::PINNED_HOST)
        );
    }
    return std::unique_ptr<Buffer>(new Buffer(std::move(data), MemoryType::DEVICE));
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    if (reservation.mem_type_ != buffer->mem_type()) {
        auto const nbytes = buffer->size;
        auto ret = make_buffer(nbytes, buffer->stream(), reservation);
        buffer_copy(statistics_, *ret, *buffer, nbytes);
        return ret;
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type_ == MemoryType::DEVICE,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    auto stream = buffer->stream();
    auto ret = move(std::move(buffer), reservation)->release_device_buffer();
    RAPIDSMPF_EXPECTS(
        ret->stream().value() == stream.value(),
        "something went wrong, the Buffer's stream and the device_buffer's stream "
        "don't match"
    );
    return ret;
}

std::unique_ptr<HostBuffer> BufferResource::move_to_host_buffer(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type_ == MemoryType::HOST,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    return move(std::move(buffer), reservation)->release_host_buffer();
}

std::shared_ptr<rmm::cuda_stream_pool> const& BufferResource::stream_pool() const {
    return stream_pool_;
}

SpillManager& BufferResource::spill_manager() {
    return spill_manager_;
}

std::shared_ptr<Statistics> BufferResource::statistics() const noexcept {
    return statistics_;
}

std::int64_t device_limit_from_options(config::Options options) {
    return options.get<std::int64_t>("spill_device_limit", [](auto const& s) {
        auto const [_, total_mem] = rmm::available_device_memory();
        return rmm::align_down(
            parse_nbytes_or_percent(s, total_mem), rmm::CUDA_ALLOCATION_ALIGNMENT
        );
    });
}

std::optional<Duration> periodic_spill_check_from_options(config::Options options) {
    return options.get<std::optional<Duration>>(
        "periodic_spill_check", [](auto const& s) -> std::optional<Duration> {
            if (auto val = parse_optional(s); val.has_value()) {
                return parse_duration(val.value());
            }
            return std::nullopt;
        }
    );
}

std::shared_ptr<rmm::cuda_stream_pool> stream_pool_from_options(config::Options options) {
    auto const num_streams =
        options.get<std::size_t>("num_streams", parse_string<std::size_t>);
    RAPIDSMPF_EXPECTS(
        num_streams > 0,
        "The `num_streams` option must be greater than 0",
        std::invalid_argument
    );
    return std::make_shared<rmm::cuda_stream_pool>(
        num_streams, rmm::cuda_stream::flags::non_blocking
    );
}

}  // namespace rapidsmpf
