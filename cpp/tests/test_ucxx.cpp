/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <gtest/gtest.h>
#include <ucp/api/ucp.h>
#include <ucxx/context.h>

#include <rapidsmpf/communicator/ucxx.hpp>

#include "communicator/ucxx_internal.hpp"

namespace {

struct ProgressModeFeatureFlagsParam {
    rapidsmpf::ucxx::ProgressMode progress_mode;
    bool wakeup_enabled;
};

class ProgressModeFeatureFlagsTest
    : public ::testing::TestWithParam<ProgressModeFeatureFlagsParam> {};

TEST_P(ProgressModeFeatureFlagsTest, CreatesContextWithExpectedFeatureFlags) {
    auto const context =
        rapidsmpf::ucxx::detail::create_context(GetParam().progress_mode);
    auto const default_flags = ::ucxx::Context::defaultFeatureFlags;
    auto const expected_flags =
        GetParam().wakeup_enabled ? default_flags : default_flags & ~UCP_FEATURE_WAKEUP;

    EXPECT_EQ(context->getFeatureFlags(), expected_flags);
}

INSTANTIATE_TEST_SUITE_P(
    ProgressModes,
    ProgressModeFeatureFlagsTest,
    ::testing::Values(
        ProgressModeFeatureFlagsParam{rapidsmpf::ucxx::ProgressMode::Blocking, true},
        ProgressModeFeatureFlagsParam{rapidsmpf::ucxx::ProgressMode::Polling, false},
        ProgressModeFeatureFlagsParam{
            rapidsmpf::ucxx::ProgressMode::ThreadBlocking, true
        },
        ProgressModeFeatureFlagsParam{rapidsmpf::ucxx::ProgressMode::ThreadPolling, false}
    )
);

}  // namespace
