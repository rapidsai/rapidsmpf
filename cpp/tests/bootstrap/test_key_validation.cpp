/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/backend.hpp>
#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/bootstrap/socket_backend.hpp>
#include <rapidsmpf/bootstrap/types.hpp>

namespace {

using rapidsmpf::bootstrap::BackendType;
using rapidsmpf::bootstrap::Context;
using rapidsmpf::bootstrap::max_key_size;
using rapidsmpf::bootstrap::detail::FileBackend;
using rapidsmpf::bootstrap::detail::SocketBackend;
using rapidsmpf::bootstrap::detail::SocketServer;

struct InvalidKeyParam {
    std::string name;
    std::string key;
};

const std::vector<InvalidKeyParam> invalid_key_params{
    {"TooLong", std::string(max_key_size + 1, 'x')},
    {"Empty", ""},
    {"Space", "foo bar"},
    {"Tab", "foo\tbar"},
    {"DotDot", ".."},
    {"Slash", "foo/bar"},
    {"Backslash", "foo\\bar"},
    // Use explicit length to embed the null byte — string literal would stop at '\0'.
    {"NullByte", std::string("foo\0bar", 7)},
};

auto param_name(const ::testing::TestParamInfo<InvalidKeyParam>& info) {
    return info.param.name;
}

}  // namespace

class FileBackendKeyTest : public ::testing::TestWithParam<InvalidKeyParam> {
  protected:
    void SetUp() override {
        char tmpdir[] = "/tmp/fb_kv_test.XXXXXX";
        char* d = ::mkdtemp(tmpdir);
        ASSERT_NE(d, nullptr) << "mkdtemp failed";
        coord_dir_ = d;

        Context ctx{
            .rank = 0,
            .nranks = 1,
            .type = BackendType::FILE,
            .coord_dir = coord_dir_,
        };
        backend_ = std::make_unique<FileBackend>(std::move(ctx));
    }

    void TearDown() override {
        backend_.reset();
        // FileBackend destructor (rank 0) removes the coord directory.
    }

    std::string coord_dir_;
    std::unique_ptr<FileBackend> backend_;
};

TEST_F(FileBackendKeyTest, MaxLengthKeyIsValid) {
    EXPECT_NO_THROW(backend_->put(std::string(max_key_size, 'x'), "value"));
}

TEST_P(FileBackendKeyTest, InvalidKeyThrows) {
    auto const& key = GetParam().key;
    EXPECT_THROW(backend_->put(key, "value"), std::invalid_argument);
    EXPECT_THROW(backend_->get(key, std::chrono::seconds{0}), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidKeys, FileBackendKeyTest, ::testing::ValuesIn(invalid_key_params), param_name
);

class SocketBackendKeyTest : public ::testing::TestWithParam<InvalidKeyParam> {
  protected:
    void SetUp() override {
        server_ = std::make_unique<SocketServer>(1);
        ::setenv("RRUN_SOCKET_ADDR", server_->address().c_str(), 1);
        ::setenv("RRUN_SOCKET_TOKEN", server_->token().c_str(), 1);

        Context ctx{
            .rank = 0,
            .nranks = 1,
            .type = BackendType::SOCKET,
        };
        backend_ = std::make_unique<SocketBackend>(std::move(ctx));
    }

    void TearDown() override {
        backend_.reset();
        server_.reset();
        ::unsetenv("RRUN_SOCKET_ADDR");
        ::unsetenv("RRUN_SOCKET_TOKEN");
    }

    std::unique_ptr<SocketServer> server_;
    std::unique_ptr<SocketBackend> backend_;
};

TEST_F(SocketBackendKeyTest, MaxLengthKeyIsValid) {
    EXPECT_NO_THROW(backend_->put(std::string(max_key_size, 'x'), "value"));
}

TEST_P(SocketBackendKeyTest, InvalidKeyThrows) {
    auto const& key = GetParam().key;
    EXPECT_THROW(backend_->put(key, "value"), std::invalid_argument);
    EXPECT_THROW(backend_->get(key, std::chrono::seconds{0}), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidKeys, SocketBackendKeyTest, ::testing::ValuesIn(invalid_key_params), param_name
);
