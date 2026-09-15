/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

namespace {

using namespace disk;

std::filesystem::path test_spill_dir() {
    return spill_dir_from_options(GlobalEnvironment->options())
        .value_or(std::filesystem::temp_directory_path());
}

std::string test_path(std::string_view suffix) {
    static std::atomic<std::uint64_t> counter{0};
    auto const id = counter.fetch_add(1, std::memory_order_relaxed);
    return (test_spill_dir()
            / ("rapidsmpf-disk-test-" + std::to_string(::getpid()) + "-"
               + std::to_string(id) + "-" + std::string{suffix} + ".bin"))
        .string();
}

std::vector<std::byte> make_pattern(std::size_t size) {
    std::vector<std::byte> result(size);
    for (std::size_t i = 0; i < size; ++i) {
        auto const value =
            static_cast<unsigned char>(((i * 131U) ^ (i >> 7U) ^ 0x5aU) & 0xffU);
        result[i] = static_cast<std::byte>(value);
    }
    return result;
}

void check_file_contents(
    std::filesystem::path const& path,
    std::size_t file_offset,
    std::vector<std::byte> const& expected
) {
    std::vector<std::byte> actual(expected.size());
    std::ifstream file{path, std::ios::binary};
    ASSERT_TRUE(file.good());
    file.seekg(static_cast<std::streamoff>(file_offset));
    file.read(
        reinterpret_cast<char*>(actual.data()),
        static_cast<std::streamsize>(actual.size())
    );
    EXPECT_EQ(file.gcount(), static_cast<std::streamsize>(actual.size()));
    EXPECT_EQ(actual, expected);
}

class DiskResourceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
            GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
        }
        br_ = BufferResource::create(
            rmm::mr::get_current_device_resource_ref(),
            PinnedMemoryDisabled,
            {},
            std::chrono::milliseconds{1},
            std::make_shared<StreamPool>(16),
            Statistics::disabled(),
            test_spill_dir()
        );
        disk_ = br_->disk_resource();
    }

    std::shared_ptr<DiskResource> disk_;
    std::shared_ptr<BufferResource> br_;
};

TEST_F(DiskResourceTest, RoundTrip) {
    auto const path = test_path("roundtrip");
    auto const pattern = make_pattern(64 * 1024);

    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), MemoryType::HOST),
        pattern.size()
    );

    std::vector<std::byte> destination(pattern.size());
    EXPECT_EQ(
        disk_->read(path, destination.data(), pattern.size(), MemoryType::HOST),
        pattern.size()
    );

    EXPECT_EQ(destination, pattern);
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_F(DiskResourceTest, UnalignedOffsetRoundTrip) {
    auto const path = test_path("unaligned");
    auto const pattern = make_pattern(16 * 1024);
    auto const ptr_offset = std::size_t{1};
    auto const file_offset = std::size_t{1};

    std::vector<std::byte> source(ptr_offset);
    source.insert(source.end(), pattern.begin(), pattern.end());

    EXPECT_EQ(
        disk_->write(
            path,
            source.data() + ptr_offset,
            pattern.size(),
            MemoryType::HOST,
            file_offset
        ),
        pattern.size()
    );

    EXPECT_EQ(std::filesystem::file_size(path), pattern.size() + file_offset);
    check_file_contents(path, file_offset, pattern);

    std::vector<std::byte> destination(pattern.size() + ptr_offset);
    EXPECT_EQ(
        disk_->read(
            path,
            destination.data() + ptr_offset,
            pattern.size(),
            MemoryType::HOST,
            file_offset
        ),
        pattern.size()
    );
    EXPECT_TRUE(
        std::ranges::equal(
            destination.begin() + ptr_offset,
            destination.end(),
            pattern.begin(),
            pattern.end()
        )
    );

    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_F(DiskResourceTest, FlushDoesNotThrow) {
    auto const path = test_path("flush");
    auto const pattern = make_pattern(4096);
    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), MemoryType::HOST),
        pattern.size()
    );
    EXPECT_NO_THROW(disk_->flush(path));
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST(DiskSpillDirectory, UnsetOptionIsEmpty) {
    config::Options options;
    EXPECT_EQ(spill_dir_from_options(options), std::nullopt);
}

TEST(DiskSpillDirectory, DisabledOptionIsEmpty) {
    config::Options options{{{"disk_spill_dir", config::OptionValue("false")}}};
    EXPECT_EQ(spill_dir_from_options(options), std::nullopt);
}

TEST(DiskSpillDirectory, EmptyStringUsesDefault) {
    config::Options options{{{"disk_spill_dir", config::OptionValue("")}}};
    EXPECT_EQ(spill_dir_from_options(options), std::nullopt);
}

TEST(DiskSpillDirectory, WhitespaceOnlyThrows) {
    config::Options options{{{"disk_spill_dir", config::OptionValue("   ")}}};
    EXPECT_THROW(std::ignore = spill_dir_from_options(options), std::invalid_argument);
}

TEST(DiskSpillDirectory, UsesConfiguredPath) {
    config::Options options{
        {{"disk_spill_dir", config::OptionValue("/tmp/rapidsmpf-spill")}}
    };
    EXPECT_EQ(
        spill_dir_from_options(options), std::filesystem::path{"/tmp/rapidsmpf-spill"}
    );
}

TEST(DiskResource, CreateWithoutSpillDirHasNoDiskResource) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    EXPECT_EQ(br->disk_resource(), nullptr);
}

TEST(DiskResource, SharedPtrKeepsDiskResourceAlive) {
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {},
        std::chrono::milliseconds{1},
        std::make_shared<StreamPool>(16),
        Statistics::disabled(),
        test_spill_dir()
    );
    std::weak_ptr<DiskResource> weak_disk = br->disk_resource();
    std::weak_ptr<BufferResource> weak_br = br;

    {
        auto disk = br->disk_resource();
        br.reset();
        EXPECT_TRUE(weak_br.expired()) << "BR should not be kept alive by DiskResource";
        EXPECT_FALSE(weak_disk.expired())
            << "DiskResource freed while a shared_ptr still holds it";
    }

    EXPECT_TRUE(weak_disk.expired()) << "DiskResource not destructed, refcount cycle?";
}

TEST_F(DiskResourceTest, DiskBufferDestructorRemovesFile) {
    std::filesystem::path path{};
    {
        DiskBuffer disk_buffer{disk_};
        path = disk_buffer.path();
        ASSERT_TRUE(std::filesystem::exists(path));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(DiskResourceTest, DiskBufferReportsFileSize) {
    DiskBuffer disk_buffer{disk_};
    EXPECT_EQ(disk_buffer.file_size(), 0U);
    EXPECT_TRUE(disk_buffer.copy_to_uint8_vector().empty());

    auto const pattern = make_pattern(1024);

    EXPECT_EQ(
        disk_->write(
            disk_buffer.path(), pattern.data(), pattern.size(), MemoryType::HOST
        ),
        pattern.size()
    );

    EXPECT_EQ(disk_buffer.file_size(), pattern.size());
    EXPECT_TRUE(
        std::ranges::equal(
            disk_buffer.copy_to_uint8_vector(),
            pattern,
            [](std::uint8_t lhs, std::byte rhs) {
                return lhs == std::to_integer<std::uint8_t>(rhs);
            }
        )
    );
}

TEST(DiskBufferConfiguredDirectory, UsesBufferResourceDirectory) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
    }

    TempDir disk_dir;
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {},
        std::chrono::milliseconds{1},
        std::make_shared<StreamPool>(16),
        Statistics::disabled(),
        disk_dir.path()
    );

    EXPECT_EQ(
        br->disk_resource()->directory(), disk_dir.path() / std::to_string(::getpid())
    );
}

TEST(DiskBufferFromOptions, UsesConfiguredDirectory) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
    }

    TempDir disk_dir;
    config::Options options{
        {{"disk_spill_dir", config::OptionValue(disk_dir.path().string())}}
    };
    auto br = BufferResource::from_options(
        rmm::mr::get_current_device_resource_ref(), std::move(options)
    );
    EXPECT_EQ(
        br->disk_resource()->directory(), disk_dir.path() / std::to_string(::getpid())
    );
}

TEST(DiskBufferFromOptions, UnsetOptionHasNoDiskResource) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
    }

    auto br = BufferResource::from_options(
        rmm::mr::get_current_device_resource_ref(), config::Options{}
    );
    EXPECT_EQ(br->disk_resource(), nullptr);
}

}  // namespace
