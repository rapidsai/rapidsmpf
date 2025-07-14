/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <rapidsmpf/communicator/single.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

void Environment::SetUp() {
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};
    comm_ = std::make_shared<rapidsmpf::Single>(options);
    split_comm_ = comm_;
    progress_thread_ = std::make_shared<rapidsmpf::ProgressThread>(comm_->logger());
}

void Environment::TearDown() {
    comm_ = nullptr;
    split_comm_ = nullptr;
}

void Environment::barrier() {}

std::shared_ptr<rapidsmpf::Communicator> Environment::split_comm() {
    return split_comm_;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
