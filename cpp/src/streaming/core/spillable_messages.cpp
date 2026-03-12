/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/spillable_messages.hpp>

namespace rapidsmpf::streaming {

SpillableMessages::MessageId SpillableMessages::insert(Message&& message) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    content_descriptions_.insert({counter_, message.content_description()});
    items_.insert({counter_, std::make_shared<Item>(std::move(message))});
    return counter_++;
}

Message SpillableMessages::extract(MessageId mid) {
    std::unique_lock global_lock(global_mutex_);
    std::shared_ptr<Item> item = extract_item(items_, mid).second;
    content_descriptions_.erase(mid);
    global_lock.unlock();

    // If the item is being spilled, we block here until the spilling is done.
    std::unique_lock item_lock(item->mutex);
    return std::exchange(item->message, std::nullopt).value();
}

Message SpillableMessages::copy(MessageId mid, MemoryReservation& reservation) {
    // Find item, if it exist.
    std::unique_lock global_lock(global_mutex_);
    auto item_it = items_.find(mid);
    RAPIDSMPF_EXPECTS(
        item_it != items_.end(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    std::shared_ptr<Item> item = item_it->second;
    global_lock.unlock();

    // Acquire the item's lock and verify that it still holds a message,
    // since it may have been extracted while the global lock was released.
    std::unique_lock item_lock(item->mutex);
    RAPIDSMPF_EXPECTS(
        item->message.has_value(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    return item->message->copy(reservation);
}

std::size_t SpillableMessages::spill(MessageId mid, BufferResource* br) const {
    // Find item, if it exist.
    std::unique_lock global_lock(global_mutex_);
    auto item_it = items_.find(mid);
    if (item_it == items_.end()) {
        return 0;
    }
    std::shared_ptr<Item> item = item_it->second;
    global_lock.unlock();

    // Acquire the item's lock and verify that it still holds a message,
    // since it may have been extracted while the global lock was released.
    std::unique_lock item_lock(item->mutex, std::try_to_lock);
    if (!item_lock.owns_lock()) {
        return 0;
    }
    if (!item->message.has_value()) {
        return 0;
    }

    // Ensure the item still contains something to spill.
    auto const& msg = item->message.value();
    auto const old_cd = msg.content_description();
    if (!old_cd.spillable() || old_cd.content_size(MemoryType::DEVICE) == 0) {
        return 0;
    }

    // Spill item in-place.
    auto res = br->reserve_or_fail(msg.copy_cost(), SPILL_TARGET_MEMORY_TYPES);
    item->message = msg.copy(res);
    auto const new_cd = item->message.value().content_description();
    item_lock.unlock();

    // Update the content descriptions only if `mid` still exists.
    // This handles the case where it may have been extracted while the item lock
    // was released and simultaneously `extract` acquired global/item locks and
    // released the item.
    std::scoped_lock lock(global_lock);
    if (auto it = content_descriptions_.find(mid); it != content_descriptions_.end()) {
        it->second = new_cd;
    }
    return old_cd.content_size(MemoryType::DEVICE);
}

std::map<SpillableMessages::MessageId, ContentDescription>
SpillableMessages::get_content_descriptions() const {
    std::unique_lock global_lock(global_mutex_);
    return content_descriptions_;
}

ContentDescription rapidsmpf::streaming::SpillableMessages::get_content_description(
    MessageId mid
) const {
    std::lock_guard global_lock(global_mutex_);
    auto it = content_descriptions_.find(mid);
    RAPIDSMPF_EXPECTS(
        it != content_descriptions_.end(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    return it->second;
}

}  // namespace rapidsmpf::streaming
