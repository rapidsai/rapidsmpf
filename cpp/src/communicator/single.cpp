/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {


Single::Single(config::Options options) : logger_{this, std::move(options)} {}

std::unique_ptr<Communicator::Future> Single::send(
    std::unique_ptr<std::vector<uint8_t>>, Rank, Tag
) {
    RAPIDSMPF_FAIL("Unexpected send to self", std::runtime_error);
}

std::unique_ptr<Communicator::Future> Single::send(std::unique_ptr<Buffer>, Rank, Tag) {
    RAPIDSMPF_FAIL("Unexpected send to self", std::runtime_error);
}

std::unique_ptr<Communicator::Future> Single::recv(Rank, Tag, std::unique_ptr<Buffer>) {
    RAPIDSMPF_FAIL("Unexpected recv from self", std::runtime_error);
}

std::unique_ptr<Communicator::Future> Single::recv_sync_host_data(
    Rank, Tag, std::unique_ptr<std::vector<uint8_t>>
) {
    RAPIDSMPF_FAIL("Unexpected recv from self", std::runtime_error);
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> Single::recv_any(Tag) {
    return {nullptr, 0};
}

std::unique_ptr<std::vector<uint8_t>> Single::recv_from(Rank, Tag) {
    return nullptr;
}

std::pair<std::vector<std::unique_ptr<Communicator::Future>>, std::vector<std::size_t>>
Single::test_some(std::vector<std::unique_ptr<Communicator::Future>>&) {
    RAPIDSMPF_FAIL("Unexpected test_some from self", std::runtime_error);
}

std::vector<std::size_t> Single::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
) {
    RAPIDSMPF_FAIL("Unexpected test_some from self", std::runtime_error);
}

bool Single::test(std::unique_ptr<Communicator::Future>&) {
    RAPIDSMPF_FAIL("Unexpected test from self", std::runtime_error);
}

std::vector<std::unique_ptr<Buffer>> Single::wait_all(
    std::vector<std::unique_ptr<Communicator::Future>>&&
) {
    RAPIDSMPF_FAIL("Unexpected wait_all from self", std::runtime_error);
}

std::unique_ptr<Buffer> Single::wait(std::unique_ptr<Communicator::Future>) {
    RAPIDSMPF_FAIL("Unexpected wait from self", std::runtime_error);
}

std::unique_ptr<Buffer> Single::release_data(std::unique_ptr<Communicator::Future>) {
    RAPIDSMPF_FAIL("Unexpected release_data from self", std::runtime_error);
}

std::unique_ptr<std::vector<uint8_t>> Single::release_sync_host_data(
    std::unique_ptr<Communicator::Future>
) {
    RAPIDSMPF_FAIL("Unexpected release_sync_host_data from self", std::runtime_error);
}

std::string Single::str() const {
    std::stringstream ss;
    ss << "Uni(rank=0, nranks: 1)";
    return ss.str();
}
}  // namespace rapidsmpf
