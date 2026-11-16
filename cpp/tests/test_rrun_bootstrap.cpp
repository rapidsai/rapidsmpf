/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Tests for bootstrapping communicators under rrun.
 *
 * These tests must be run with rrun, e.g.:
 *   rrun -n 2 gtests/rrun_tests --gtest_filter="*RrunBootstrap*"
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <rapidsmpf/bootstrap/utils.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX
#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/progress_thread.hpp>
#endif

class RrunBootstrapTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifndef RAPIDSMPF_HAVE_UCXX
        GTEST_SKIP() << "rapidsmpf was built without UCXX support";
#endif
        if (!rapidsmpf::bootstrap::is_running_with_rrun()) {
            GTEST_SKIP() << "Test must be run with rrun (RRUN_RANK not set)";
        }
    }
};

#ifdef RAPIDSMPF_HAVE_UCXX

namespace {

std::shared_ptr<rapidsmpf::ucxx::UCXX> bootstrap_ucxx_comm() {
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};
    return rapidsmpf::bootstrap::create_ucxx_comm(
        std::make_shared<rapidsmpf::ProgressThread>(),
        rapidsmpf::bootstrap::BackendType::AUTO,
        options,
        rapidsmpf::Logger::from_options(options)
    );
}

}  // namespace

// Bootstrapping several communicators in one process works, and earlier ones
// stay usable while later ones are created and used.
TEST_F(RrunBootstrapTest, RepeatedCreateUcxxComm) {
    auto const rank = rapidsmpf::bootstrap::get_rank();
    auto const nranks = rapidsmpf::bootstrap::get_nranks();
    std::vector<std::shared_ptr<rapidsmpf::ucxx::UCXX>> comms;
    for (int i = 0; i < 3; ++i) {
        comms.push_back(bootstrap_ucxx_comm());
        auto const& comm = comms.back();
        EXPECT_EQ(comm->nranks(), nranks);
        // The bootstrap root is always UCXX rank 0. Other ranks are numbered in
        // the order they connect, which can differ between communicators.
        EXPECT_EQ(comm->rank() == 0, rank == 0);
        comm->barrier();
        comms.front()->barrier();
    }
    // Short-lived communicators can be dropped while the first one stays in use.
    comms.resize(1);
    comms.front()->barrier();
}

#endif  // RAPIDSMPF_HAVE_UCXX
