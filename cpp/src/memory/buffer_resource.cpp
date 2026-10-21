/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

namespace {

rapidsmpf_string_view string_view(std::string const& value) {
    return {value.data(), value.size()};
}

struct AttributeViews {
    std::vector<rapidsmpf_reservation_attribute> values;

    explicit AttributeViews(ReservationAttributes const& attributes) {
        if (!attributes) {
            return;
        }
        values.reserve(attributes->size());
        for (auto const& [key, value] : *attributes) {
            rapidsmpf_reservation_attribute item{};
            item.key = string_view(key);
            std::visit(
                [&item](auto const& entry) {
                    using T = std::decay_t<decltype(entry)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        item.type = RAPIDSMPF_RESERVATION_ATTRIBUTE_STRING;
                        item.value.string_value = string_view(entry);
                    } else if constexpr (std::is_same_v<T, std::int64_t>) {
                        item.type = RAPIDSMPF_RESERVATION_ATTRIBUTE_I64;
                        item.value.i64_value = entry;
                    } else if constexpr (std::is_same_v<T, bool>) {
                        item.type = RAPIDSMPF_RESERVATION_ATTRIBUTE_BOOL;
                        item.value.bool_value = entry;
                    } else {
                        item.type = RAPIDSMPF_RESERVATION_ATTRIBUTE_F64;
                        item.value.f64_value = entry;
                    }
                },
                value
            );
            values.push_back(item);
        }
    }

    operator rapidsmpf_reservation_attributes_view() const {
        return {values.data(), values.size()};
    }
};

}  // namespace

class BufferResource::RecorderRegistration {
  public:
    explicit RecorderRegistration(rapidsmpf_reservation_recorder const& recorder)
        : recorder_{recorder} {
        if (recorder_.retain_user != nullptr) {
            recorder_.retain_user(recorder_.user);
        }
    }

    ~RecorderRegistration() {
        if (recorder_.release_user != nullptr) {
            recorder_.release_user(recorder_.user);
        }
    }

    template <typename F>
    void invoke(F&& callback) noexcept {
        std::unique_lock lock(mutex_);
        if (!accepting_) {
            return;
        }
        ++in_flight_;
        auto const sequence = sequence_++;
        lock.unlock();
        callback(recorder_, sequence);
        lock.lock();
        if (--in_flight_ == 0) {
            drained_.notify_all();
        }
    }

    void drain() noexcept {
        std::unique_lock lock(mutex_);
        accepting_ = false;
        drained_.wait(lock, [this] { return in_flight_ == 0; });
    }

  private:
    rapidsmpf_reservation_recorder recorder_;
    std::mutex mutex_;
    std::condition_variable drained_;
    std::uint64_t sequence_{0};
    std::size_t in_flight_{0};
    bool accepting_{true};
};

namespace {
template <typename F>
void emit(
    std::mutex& mutex,
    std::shared_ptr<BufferResource::RecorderRegistration> const& recorder,
    F&& callback
) {
    std::shared_ptr<BufferResource::RecorderRegistration> registration;
    {
        std::lock_guard lock(mutex);
        registration = recorder;
    }
    if (registration) {
        registration->invoke(std::forward<F>(callback));
    }
}
}  // namespace

std::uint64_t BufferResource::next_reservation_id() noexcept {
    return reservation_id_counter_.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t BufferResource::next_request_id() noexcept {
    return request_id_counter_.fetch_add(1, std::memory_order_relaxed);
}

void BufferResource::emit_request_submitted(
    std::uint64_t id,
    std::size_t size,
    std::span<MemoryType const> memory_types,
    rapidsmpf_reservation_request_policy policy,
    std::optional<std::int64_t> delta,
    ReservationAttributes const& attributes
) {
    std::vector<std::uint32_t> types;
    types.reserve(memory_types.size());
    for (auto type : memory_types) {
        types.push_back(static_cast<std::uint32_t>(type));
    }
    AttributeViews attrs{attributes};
    emit(recorder_mutex_, recorder_, [&](auto const& recorder, std::uint64_t sequence) {
        if (recorder.request_submitted != nullptr) {
            recorder.request_submitted(
                recorder.user,
                sequence,
                id,
                size,
                types.data(),
                types.size(),
                policy,
                delta.has_value(),
                delta.value_or(0),
                attrs
            );
        }
    });
}

void BufferResource::emit_request_waiting(std::uint64_t id) {
    emit(recorder_mutex_, recorder_, [=](auto const& recorder, std::uint64_t sequence) {
        if (recorder.request_waiting != nullptr) {
            recorder.request_waiting(recorder.user, sequence, id);
        }
    });
}

void BufferResource::emit_request_resolved(
    std::uint64_t id,
    rapidsmpf_reservation_request_outcome outcome,
    detail::ReserveResult const* result,
    bool timed_out,
    std::string const& error
) {
    emit(recorder_mutex_, recorder_, [&](auto const& recorder, std::uint64_t sequence) {
        if (recorder.request_resolved != nullptr) {
            recorder.request_resolved(
                recorder.user,
                sequence,
                id,
                outcome,
                result == nullptr ? 0 : result->reservation.id(),
                result == nullptr ? 0 : static_cast<std::uint32_t>(result->mem_type),
                result == nullptr ? 0 : result->reservation.size(),
                result == nullptr ? 0 : result->overbooked_bytes,
                timed_out,
                string_view(error)
            );
        }
    });
}

void BufferResource::emit_reservation_created(
    MemoryReservation const& reservation,
    std::uint64_t request_id,
    std::uint64_t parent_id
) {
    AttributeViews attrs{reservation.attributes()};
    emit(recorder_mutex_, recorder_, [&](auto const& recorder, std::uint64_t sequence) {
        if (recorder.reservation_created != nullptr) {
            recorder.reservation_created(
                recorder.user,
                sequence,
                reservation.id(),
                static_cast<std::uint32_t>(reservation.mem_type()),
                reservation.size(),
                attrs,
                request_id,
                parent_id
            );
        }
    });
}

void BufferResource::emit_reservation_resized(
    std::uint64_t id, std::size_t size, rapidsmpf_reservation_resize_reason reason
) {
    emit(recorder_mutex_, recorder_, [=](auto const& recorder, std::uint64_t sequence) {
        if (recorder.reservation_resized != nullptr) {
            recorder.reservation_resized(recorder.user, sequence, id, size, reason);
        }
    });
}

void BufferResource::emit_reservation_released(
    std::uint64_t id, rapidsmpf_reservation_release_reason reason
) {
    if (id == 0) {
        return;
    }
    emit(recorder_mutex_, recorder_, [=](auto const& recorder, std::uint64_t sequence) {
        if (recorder.reservation_released != nullptr) {
            recorder.reservation_released(recorder.user, sequence, id, reason);
        }
    });
}

namespace detail {

RequestScope::RequestScope(
    BufferResource& br,
    std::size_t size,
    std::span<MemoryType const> memory_types,
    rapidsmpf_reservation_request_policy policy,
    std::optional<std::int64_t> net_memory_delta,
    ReservationAttributes attributes
)
    : br_{&br}, id_{br.next_request_id()}, attributes_{std::move(attributes)} {
    br_->emit_request_submitted(
        id_, size, memory_types, policy, net_memory_delta, attributes_
    );
}

RequestScope::~RequestScope() noexcept {
    assert(resolved_ && "reservation request left unresolved");
}

void RequestScope::waiting() {
    if (!waiting_emitted_) {
        waiting_emitted_ = true;
        br_->emit_request_waiting(id_);
    }
}

void RequestScope::resolve(
    rapidsmpf_reservation_request_outcome outcome,
    ReserveResult const* result,
    bool timed_out,
    std::string const& error
) {
    assert(!resolved_ && "reservation request resolved more than once");
    if (resolved_) {
        return;
    }
    resolved_ = true;
    br_->emit_request_resolved(id_, outcome, result, timed_out, error);
}

void RequestScope::resolve_granted(ReserveResult const& result) {
    resolve(RAPIDSMPF_REQUEST_GRANTED, &result, result.timed_out, {});
}

void RequestScope::resolve_empty(bool timed_out) {
    resolve(RAPIDSMPF_REQUEST_EMPTY, nullptr, timed_out, {});
}

void RequestScope::resolve_failed(std::string const& error) {
    resolve(RAPIDSMPF_REQUEST_FAILED, nullptr, false, error);
}

void RequestScope::resolve_cancelled() {
    resolve(RAPIDSMPF_REQUEST_CANCELLED, nullptr, false, {});
}

}  // namespace detail

BufferResource::BufferResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
    std::optional<PinnedMemoryResource> pinned_mr,
    std::unordered_map<MemoryType, std::int64_t> memory_limits,
    std::optional<Duration> periodic_spill_check,
    std::shared_ptr<StreamPool> stream_pool,
    std::shared_ptr<Statistics> statistics,
    std::shared_ptr<DiskResource> disk_resource,
    Duration memory_reserve_timeout,
    AllowOverbooking allow_overbooking_by_default
)
    : owning_mr_{std::move(device_mr)},
      pinned_mr_{std::move(pinned_mr)},
      host_mr_{},
      disk_resource_{std::move(disk_resource)},
      stream_pool_{std::move(stream_pool)},
      spill_manager_{this, periodic_spill_check},
      statistics_{std::move(statistics)},
      memory_reserve_timeout_{memory_reserve_timeout},
      allow_overbooking_by_default_{allow_overbooking_by_default} {
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

BufferResource::~BufferResource() noexcept {
    set_reservation_recorder(nullptr);
}

std::shared_ptr<BufferResource> BufferResource::create(
    cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
    std::optional<PinnedPoolProperties> pinned_pool_properties,
    std::unordered_map<MemoryType, std::int64_t> memory_limits,
    std::optional<Duration> periodic_spill_check,
    std::shared_ptr<StreamPool> stream_pool,
    std::shared_ptr<Statistics> statistics,
    std::optional<std::filesystem::path> spill_directory,
    Duration memory_reserve_timeout,
    AllowOverbooking allow_overbooking_by_default
) {
    std::optional<PinnedMemoryResource> pinned_mr;
    if (pinned_pool_properties.has_value()) {
        RAPIDSMPF_EXPECTS(
            !pinned_pool_properties->max_pool_size.has_value()
                || *pinned_pool_properties->max_pool_size > 0,
            "PinnedPoolProperties::max_pool_size must be greater than zero",
            std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            is_pinned_memory_resources_supported(),
            "pinned host memory was requested (via `PinnedPoolProperties`) but is not "
            "supported on this system. "
            "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
            " is one of the requirements, but additional platform or driver constraints "
            "may apply. Pass `PinnedMemoryDisabled` to disable pinned host memory.",
            std::runtime_error
        );
        pinned_mr = PinnedMemoryResource{*pinned_pool_properties};
    }

    std::shared_ptr<DiskResource> disk_res;
    if (spill_directory.has_value()) {
        disk_res.reset(new DiskResource{*spill_directory / std::to_string(::getpid())});
    }
    std::shared_ptr<BufferResource> br{new BufferResource{
        std::move(device_mr),
        std::move(pinned_mr),
        std::move(memory_limits),
        periodic_spill_check,
        std::move(stream_pool),
        std::move(statistics),
        std::move(disk_res),
        memory_reserve_timeout,
        allow_overbooking_by_default
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
    if (br->disk_resource_ != nullptr) {
        br->disk_resource_->set_backref(weak);
    }
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
    auto pinned_pool_properties = pinned_pool_properties_from_options(options);
    auto memory_limits = memory_limits_from_options(options, pinned_pool_properties);
    return create(
        std::move(mr),
        std::move(pinned_pool_properties),
        std::move(memory_limits),
        periodic_spill_check_from_options(options),
        stream_pool_from_options(options),
        std::move(statistics),
        spill_dir_from_options(options),
        memory_reserve_timeout_from_options(options),
        allow_overbooking_by_default_from_options(options)
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
        return limit - host_mr_.current_allocated();
    case MemoryType::DISK:
        return disk_resource_ != nullptr ? limit : 0;
    }
    return std::numeric_limits<std::int64_t>::max();
}

std::int64_t BufferResource::memory_limit(MemoryType mem_type) const noexcept {
    return memory_limits_[static_cast<std::size_t>(mem_type)].load(
        std::memory_order_acquire
    );
}

bool BufferResource::has_memory_resource(MemoryType mem_type) const noexcept {
    switch (mem_type) {
    case MemoryType::PINNED_HOST:
        return pinned_mr_.has_value();
    case MemoryType::DISK:
        return disk_resource_ != nullptr;
    default:
        return true;
    }
}

void BufferResource::set_memory_limit(MemoryType mem_type, std::int64_t limit) noexcept {
    memory_limits_[static_cast<std::size_t>(mem_type)].store(
        limit, std::memory_order_release
    );
    emit(recorder_mutex_, recorder_, [=](auto const& recorder, std::uint64_t sequence) {
        if (recorder.memory_limit_changed != nullptr) {
            recorder.memory_limit_changed(
                recorder.user, sequence, static_cast<std::uint32_t>(mem_type), limit
            );
        }
    });
}

void BufferResource::set_reservation_recorder(
    rapidsmpf_reservation_recorder const* recorder
) {
    std::shared_ptr<RecorderRegistration> replacement;
    if (recorder != nullptr) {
        RAPIDSMPF_EXPECTS(
            recorder->abi_version == RAPIDSMPF_RESERVATION_RECORDER_ABI_VERSION,
            "unsupported reservation recorder ABI version",
            std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            recorder->struct_size >= sizeof(rapidsmpf_reservation_recorder),
            "reservation recorder struct is too small",
            std::invalid_argument
        );
        replacement = std::make_shared<RecorderRegistration>(*recorder);
    }
    std::shared_ptr<RecorderRegistration> previous;
    {
        std::lock_guard lock(recorder_mutex_);
        previous = std::exchange(recorder_, std::move(replacement));
    }
    if (previous) {
        previous->drain();
    }
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

std::int64_t BufferResource::memory_available_for_reservation(MemoryType mem_type) const {
    // Sample availability before taking the lock, matching `reserve()`.
    std::int64_t const available = memory_available(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    return available
           - safe_cast<std::int64_t>(
               memory_reserved_[static_cast<std::size_t>(mem_type)]
           );
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, AllowOverbooking allow_overbooking
) {
    return reserve(mem_type, size, allow_overbooking, {});
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type,
    std::size_t size,
    AllowOverbooking allow_overbooking,
    ReservationAttributes attributes
) {
    if (size == 0) {
        return {MemoryReservation(mem_type, this, 0), 0};
    }
    std::array const types{mem_type};
    detail::RequestScope scope{
        *this,
        size,
        types,
        allow_overbooking == AllowOverbooking::YES ? RAPIDSMPF_REQUEST_OVERBOOK
                                                   : RAPIDSMPF_REQUEST_IMMEDIATE,
        std::nullopt,
        std::move(attributes)
    };
    try {
        auto result = reserve_impl(mem_type, size, allow_overbooking, &scope);
        if (result.reservation.size() == 0) {
            scope.resolve_empty(false);
        } else {
            scope.resolve_granted(result);
        }
        return {std::move(result.reservation), result.total_overbooked_bytes};
    } catch (std::exception const& e) {
        scope.resolve_failed(e.what());
        throw;
    } catch (...) {
        scope.resolve_failed("unknown exception");
        throw;
    }
}

detail::ReserveResult BufferResource::reserve_impl(
    MemoryType mem_type,
    std::size_t size,
    AllowOverbooking allow_overbooking,
    detail::RequestScope* request
) {
    RAPIDSMPF_EXPECTS(
        mem_type != MemoryType::DISK || disk_resource_ != nullptr,
        "disk memory was requested but no disk resource is available",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        mem_type != MemoryType::PINNED_HOST || pinned_mr_.has_value(),
        "pinned memory resource is not available",
        std::invalid_argument
    );

    std::int64_t const available = memory_available(mem_type);
    std::size_t overbooking;
    MemoryReservation reservation;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t& reserved = memory_reserved_[static_cast<std::size_t>(mem_type)];
        std::int64_t headroom =
            available
            - (safe_cast<std::int64_t>(reserved) + safe_cast<std::int64_t>(size));
        overbooking = headroom < 0 ? safe_cast<std::size_t>(std::abs(headroom)) : 0;
        if (overbooking > 0 && allow_overbooking == AllowOverbooking::NO) {
            return {
                MemoryReservation(mem_type, this, 0),
                mem_type,
                std::min(size, overbooking),
                overbooking
            };
        }
        reserved += size;
        reservation = MemoryReservation(
            mem_type,
            this,
            size,
            size == 0 ? 0 : next_reservation_id(),
            request == nullptr ? ReservationAttributes{} : request->attributes()
        );
    }
    if (size > 0) {
        emit_reservation_created(reservation, request == nullptr ? 0 : request->id(), 0);
    }
    return {std::move(reservation), mem_type, std::min(size, overbooking), overbooking};
}

MemoryReservation BufferResource::reserve_device_memory_and_spill(
    std::size_t size, AllowOverbooking allow_overbooking
) {
    return reserve_device_memory_and_spill(size, allow_overbooking, {});
}

MemoryReservation BufferResource::reserve_device_memory_and_spill(
    std::size_t size, AllowOverbooking allow_overbooking, ReservationAttributes attributes
) {
    if (size == 0) {
        return MemoryReservation(MemoryType::DEVICE, this, 0);
    }
    std::array const types{MemoryType::DEVICE};
    detail::RequestScope scope{
        *this, size, types, RAPIDSMPF_REQUEST_SPILL, std::nullopt, std::move(attributes)
    };
    try {
        auto result =
            reserve_impl(MemoryType::DEVICE, size, AllowOverbooking::YES, &scope);
        auto& reservation = result.reservation;
        auto const total_overbooking = result.overbooked_bytes;

        if (total_overbooking > 0) {
            auto const own_overbooking = total_overbooking;
            // With a headroom of zero, `deficit` is the overbooking still outstanding
            // once the spill lock is held. All of it is spilled, but this call is judged
            // only on its own share.
            auto const [deficit, spilled] = spill_manager_.spill_to_make_headroom();
            RAPIDSMPF_EXPECTS(
                allow_overbooking == AllowOverbooking::YES
                    || spilled >= std::min(own_overbooking, deficit),
                "failed to spill enough memory (reserved: " + format_nbytes(size)
                    + ", overbooking: " + format_nbytes(own_overbooking)
                    + ", spilled: " + format_nbytes(spilled) + ")",
                rapidsmpf::reservation_error
            );
        }
        scope.resolve_granted(result);
        return std::move(reservation);
    } catch (std::exception const& e) {
        scope.resolve_failed(e.what());
        throw;
    } catch (...) {
        scope.resolve_failed("unknown exception");
        throw;
    }
}

std::size_t BufferResource::release(MemoryReservation& reservation, std::size_t size) {
    std::size_t remaining;
    auto const id = reservation.id_;
    {
        std::lock_guard const lock(mutex_);
        RAPIDSMPF_EXPECTS(
            size <= reservation.size_,
            "MemoryReservation(" + format_nbytes(reservation.size_)
                + ") isn't big enough (" + format_nbytes(size) + ")",
            rapidsmpf::reservation_error
        );
        std::size_t& reserved =
            memory_reserved_[static_cast<std::size_t>(reservation.mem_type_)];
        RAPIDSMPF_EXPECTS(reserved >= size, "corrupted reservation stat");
        reserved -= size;
        remaining = reservation.size_ -= size;
    }
    if (size > 0) {
        if (remaining == 0) {
            emit_reservation_released(id, RAPIDSMPF_RESERVATION_RELEASED_CONSUMED);
        } else {
            emit_reservation_resized(id, remaining, RAPIDSMPF_RESERVATION_CONSUMED);
        }
    }
    return remaining;
}

void BufferResource::clear_reservation(MemoryReservation& reservation) noexcept {
    auto const id = reservation.id_;
    {
        std::lock_guard const lock(mutex_);
        auto& reserved =
            memory_reserved_[static_cast<std::size_t>(reservation.mem_type_)];
        assert(reserved >= reservation.size_);
        reserved -= reservation.size_;
        reservation.size_ = 0;
    }
    emit_reservation_released(id, RAPIDSMPF_RESERVATION_CLEARED);
}

std::unique_ptr<Buffer> BufferResource::make_buffer(
    std::size_t size, cuda::stream_ref stream, MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        stream != cuda::invalid_stream, "stream must be valid", std::invalid_argument
    );
    auto const mem_type = reservation.mem_type_;
    // A disk buffer is created with mkstemp, which never touches the stream, so a
    // stream-ordered timing would measure stream backlog rather than the creation.
    StreamOrderedTiming timing{
        stream, mem_type == MemoryType::DISK ? Statistics::disabled() : statistics_
    };
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
    case MemoryType::DISK:
        ret = std::unique_ptr<Buffer>(
            new Buffer(std::make_unique<DiskBuffer>(disk_resource_, size, stream), stream)
        );
        break;
    default:
        RAPIDSMPF_FAIL("MemoryType: unknown");
    }
    release(reservation, size);
    statistics_->record_alloc(mem_type, size, std::move(timing));
    return ret;
}

std::unique_ptr<Buffer> BufferResource::make_buffer(
    cuda::stream_ref stream, MemoryReservation&& reservation
) {
    return make_buffer(reservation.size(), stream, reservation);
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    std::shared_ptr<SpillTrackToken> spill_token
) {
    RAPIDSMPF_EXPECTS(
        stream != cuda::invalid_stream, "stream must be valid", std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        spill_token == nullptr || is_host_accessible(data->memory_resource()),
        "a spill token cannot be attached to device memory",
        std::invalid_argument
    );
    // An empty buffer owns no memory, so there is no work to join. This also covers
    // default-constructed RMM buffers, which have no stream.
    if (data->capacity() > 0) {
        cuda_stream_join(stream, data->stream());
    }
    data->set_stream(stream);

    if (is_host_accessible(data->memory_resource())) {
        auto pinned_host_buffer = std::make_unique<HostBuffer>(
            HostBuffer::from_rmm_device_buffer(std::move(data), stream)
        );
        auto ret = std::unique_ptr<Buffer>(
            new Buffer(std::move(pinned_host_buffer), stream, MemoryType::PINNED_HOST)
        );
        ret->spill_track_token_ = std::move(spill_token);
        return ret;
    }
    return std::unique_ptr<Buffer>(new Buffer(std::move(data), MemoryType::DEVICE));
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    if (reservation.mem_type_ == buffer->mem_type()) {
        return buffer;
    }
    auto const nbytes = buffer->size;
    // An empty buffer holds no capacity, so relocating it is not a spill.
    bool const tracked = nbytes > 0;
    auto const from = buffer->mem_type();
    auto token = std::move(buffer->spill_track_token_);

    // Closes the spill: `make_buffer` takes the capacity back here. A token can arrive
    // while statistics are disabled, through the device-buffer overload of `move()`, so
    // this does not check `enabled()`. `add_duration_stat` is a no-op when disabled.
    auto ret = make_buffer(nbytes, buffer->stream(), reservation);
    if (tracked && reservation.mem_type_ == MemoryType::DEVICE && token != nullptr) {
        statistics_->add_duration_stat(
            "buffer-spilled-time", Duration{Clock::now() - token->since}
        );
        token.reset();
    }

    buffer_copy(statistics_, *ret, *buffer, nbytes);

    // Opens the spill: releasing the source gives the capacity back. `buffer_copy` has
    // made the source's stream wait for the copy, so destroying it here is safe.
    buffer.reset();
    if (from == MemoryType::DEVICE) {
        // Not opened while disabled, since enabling later would close an interval
        // whose start was never observed.
        if (tracked && statistics_->enabled()) {
            ret->spill_track_token_ = std::make_shared<SpillTrackToken>();
        }
    } else {
        ret->spill_track_token_ = std::move(token);
    }
    return ret;
}

std::unique_ptr<Buffer> BufferResource::ensure_memory_type(
    std::unique_ptr<Buffer> buffer, std::span<MemoryType const> accepted
) {
    RAPIDSMPF_EXPECTS(
        buffer != nullptr, "the buffer pointer cannot be NULL", std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        !accepted.empty(),
        "the accepted memory types cannot be empty",
        std::invalid_argument
    );
    if (contains(accepted, buffer->mem_type())) {
        return buffer;
    }
    auto res = try_reserve(buffer->size, accepted | std::views::reverse);
    if (!res.has_value()) {
        RAPIDSMPF_EXPECTS(
            allow_overbooking_by_default() == AllowOverbooking::YES,
            "cannot reserve memory to move the buffer and overbooking is not allowed",
            rapidsmpf::reservation_error
        );
        res = reserve(accepted.back(), buffer->size, AllowOverbooking::YES).first;
    }
    return move(std::move(buffer), *res);
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
        ret->stream().get() == stream.get(),
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

std::shared_ptr<StreamPool> const& BufferResource::stream_pool() const {
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

std::unordered_map<MemoryType, std::int64_t> memory_limits_from_options(
    config::Options options,
    std::optional<PinnedPoolProperties> const& pinned_pool_properties
) {
    std::unordered_map<MemoryType, std::int64_t> ret{
        {MemoryType::DEVICE, device_limit_from_options(options)},
    };
    if (auto const host_limit = host_limit_from_options(options); host_limit) {
        ret[MemoryType::HOST] = *host_limit;
    }
    if (pinned_pool_properties.has_value()
        && pinned_pool_properties->max_pool_size.has_value())
    {
        ret[MemoryType::PINNED_HOST] =
            safe_cast<std::int64_t>(*pinned_pool_properties->max_pool_size);
    }
    return ret;
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

std::shared_ptr<StreamPool> stream_pool_from_options(config::Options options) {
    auto const num_streams =
        options.get<std::size_t>("num_streams", parse_string<std::size_t>);
    RAPIDSMPF_EXPECTS(
        num_streams > 0,
        "The `num_streams` option must be greater than 0",
        std::invalid_argument
    );
    return std::make_shared<StreamPool>(num_streams);
}

Duration memory_reserve_timeout_from_options(config::Options options) {
    return options.get<Duration>("memory_reserve_timeout", parse_duration);
}

AllowOverbooking allow_overbooking_by_default_from_options(config::Options options) {
    return options.get<bool>("allow_overbooking_by_default", parse_string<bool>)
               ? AllowOverbooking::YES
               : AllowOverbooking::NO;
}

}  // namespace rapidsmpf
