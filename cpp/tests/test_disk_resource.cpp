/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

namespace {

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
    std::ptrdiff_t file_offset,
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

/// Regular files directly under @p dir.
std::vector<std::filesystem::path> files_in(std::filesystem::path const& dir) {
    std::vector<std::filesystem::path> ret;
    for (auto const& entry : std::filesystem::directory_iterator{dir}) {
        if (entry.is_regular_file()) {
            ret.push_back(entry.path());
        }
    }
    return ret;
}

bool bytes_equal(
    std::vector<std::uint8_t> const& lhs, std::vector<std::byte> const& rhs
) {
    return lhs.size() == rhs.size()
           && std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
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

    /// A disk buffer on @p disk holding @p data. Its file is created through the disk
    /// resource and handed to the buffer, since the buffer's path is not exposed.
    DiskBuffer make_disk_buffer(
        std::shared_ptr<DiskResource> const& disk, std::vector<std::byte> const& data
    ) {
        auto const file = test_path("fixture");
        EXPECT_EQ(disk->write(file, data.data(), data.size(), stream_), data.size());
        return DiskBuffer{disk, file, stream_};
    }

    /// A second buffer resource, and so a second disk directory.
    std::shared_ptr<BufferResource> make_other_br() {
        return BufferResource::create(
            rmm::mr::get_current_device_resource_ref(),
            PinnedMemoryDisabled,
            {},
            std::chrono::milliseconds{1},
            std::make_shared<StreamPool>(1),
            Statistics::disabled(),
            test_spill_dir()
        );
    }

    std::shared_ptr<DiskResource> disk_;
    std::shared_ptr<BufferResource> br_;
    cuda::stream_ref stream_{cudaStreamLegacy};
};

TEST_F(DiskResourceTest, RoundTrip) {
    auto const path = test_path("roundtrip");
    auto const pattern = make_pattern(64 * 1024);

    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), stream_), pattern.size()
    );

    std::vector<std::byte> destination(pattern.size());
    EXPECT_EQ(
        disk_->read(path, destination.data(), pattern.size(), stream_), pattern.size()
    );

    EXPECT_EQ(destination, pattern);
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_F(DiskResourceTest, UnalignedOffsetRoundTrip) {
    auto const path = test_path("unaligned");
    auto const pattern = make_pattern(16 * 1024);
    auto const ptr_offset = std::size_t{1};
    auto const file_offset = std::ptrdiff_t{1};

    std::vector<std::byte> source(ptr_offset);
    source.insert(source.end(), pattern.begin(), pattern.end());

    EXPECT_EQ(
        disk_->write(
            path, source.data() + ptr_offset, pattern.size(), stream_, file_offset
        ),
        pattern.size()
    );

    EXPECT_EQ(std::filesystem::file_size(path), pattern.size() + file_offset);
    check_file_contents(path, file_offset, pattern);

    std::vector<std::byte> destination(pattern.size() + ptr_offset);
    EXPECT_EQ(
        disk_->read(
            path, destination.data() + ptr_offset, pattern.size(), stream_, file_offset
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

TEST_F(DiskResourceTest, SequentialOffsetWritesPreserveExistingBytes) {
    auto const path = test_path("two-ranges");
    auto const first = make_pattern(8 * 1024);
    auto const second = make_pattern(4 * 1024);
    auto const second_offset = std::ptrdiff_t{12 * 1024};

    EXPECT_EQ(disk_->write(path, first.data(), first.size(), stream_), first.size());
    EXPECT_EQ(
        disk_->write(path, second.data(), second.size(), stream_, second_offset),
        second.size()
    );

    EXPECT_EQ(std::filesystem::file_size(path), second_offset + second.size());
    check_file_contents(path, 0, first);
    check_file_contents(path, second_offset, second);

    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_F(DiskResourceTest, CopyPreservesBytesOutsideWrittenRange) {
    auto const src_path = test_path("copy-src");
    auto const dst_path = test_path("copy-dst");
    auto const src = make_pattern(8 * 1024);
    auto const dst = make_pattern(16 * 1024);
    constexpr std::size_t size = 2 * 1024;
    constexpr std::ptrdiff_t src_offset = 1024;
    constexpr std::ptrdiff_t dst_offset = 4 * 1024;

    EXPECT_EQ(disk_->write(src_path, src.data(), src.size(), stream_), src.size());
    EXPECT_EQ(disk_->write(dst_path, dst.data(), dst.size(), stream_), dst.size());
    EXPECT_EQ(
        disk_->copy(src_path, dst_path, size, stream_, src_offset, dst_offset), size
    );

    EXPECT_EQ(std::filesystem::file_size(dst_path), dst.size());
    auto expected = dst;
    std::copy_n(src.begin() + src_offset, size, expected.begin() + dst_offset);
    check_file_contents(dst_path, 0, expected);

    ASSERT_TRUE(std::filesystem::remove(src_path));
    ASSERT_TRUE(std::filesystem::remove(dst_path));
}

TEST_F(DiskResourceTest, CopyRejectsInvalidArguments) {
    auto const src_path = test_path("copy-invalid-src");
    auto const dst_path = test_path("copy-invalid-dst");
    auto const src = make_pattern(1024);
    EXPECT_EQ(disk_->write(src_path, src.data(), src.size(), stream_), src.size());

    // Negative offsets.
    EXPECT_THROW(
        std::ignore = disk_->copy(src_path, dst_path, 16, stream_, -1, 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        std::ignore = disk_->copy(src_path, dst_path, 16, stream_, 0, -1),
        std::invalid_argument
    );
    // Same source and destination path.
    EXPECT_THROW(
        std::ignore = disk_->copy(src_path, src_path, 16, stream_), std::invalid_argument
    );

    ASSERT_TRUE(std::filesystem::remove(src_path));
}

TEST_F(DiskResourceTest, AdoptAcrossFilesystems) {
    // Find a writable directory on a different device than the spill directory. If the
    // spill directory is itself on /dev/shm, a plain rename would succeed and the
    // fallback would not run.
    auto device_of = [](std::filesystem::path const& path) {
        struct stat st{};
        return ::stat(path.c_str(), &st) == 0 ? std::optional<dev_t>{st.st_dev}
                                              : std::nullopt;
    };
    auto const spill_device = device_of(disk_->directory());
    ASSERT_TRUE(spill_device.has_value());
    std::optional<std::filesystem::path> other_dir;
    for (auto const& candidate : std::vector<std::filesystem::path>{
             "/dev/shm", std::filesystem::temp_directory_path()
         })
    {
        if (device_of(candidate).value_or(*spill_device) != *spill_device
            && ::access(candidate.c_str(), W_OK) == 0)
        {
            other_dir = candidate;
            break;
        }
    }
    if (!other_dir.has_value()) {
        GTEST_SKIP() << "No writable directory on a different device than the spill dir";
    }

    TempDir src_dir{*other_dir};
    auto const src_path = src_dir.path() / "src";
    auto const src = make_pattern(8 * 1024);
    EXPECT_EQ(disk_->write(src_path, src.data(), src.size(), stream_), src.size());

    auto const adopted = disk_->adopt(src_path);

    EXPECT_FALSE(std::filesystem::exists(src_path));
    EXPECT_EQ(adopted.parent_path(), disk_->directory());
    EXPECT_EQ(std::filesystem::file_size(adopted), src.size());
    check_file_contents(adopted, 0, src);
    // No temporary file is left next to the adopted one.
    EXPECT_EQ(
        std::distance(
            std::filesystem::directory_iterator{disk_->directory()},
            std::filesystem::directory_iterator{}
        ),
        1
    );

    ASSERT_TRUE(std::filesystem::remove(adopted));
}

TEST_F(DiskResourceTest, FlushDoesNotThrow) {
    auto const path = test_path("flush");
    auto const pattern = make_pattern(4096);
    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), stream_), pattern.size()
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
    EXPECT_FALSE(br->is_reservable(MemoryType::DISK));
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
    EXPECT_TRUE(br->is_reservable(MemoryType::DISK));
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

TEST(DiskResource, DestructorRemovesDirectory) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
    }

    TempDir disk_dir;
    std::filesystem::path resource_dir;
    {
        auto br = BufferResource::create(
            rmm::mr::get_current_device_resource_ref(),
            PinnedMemoryDisabled,
            {},
            std::chrono::milliseconds{1},
            std::make_shared<StreamPool>(16),
            Statistics::disabled(),
            disk_dir.path()
        );
        resource_dir = br->disk_resource()->directory();
        EXPECT_EQ(resource_dir.parent_path(), disk_dir.path());
    }
    EXPECT_FALSE(std::filesystem::exists(resource_dir));
}

TEST(DiskResource, ResourcesOwnDistinctDirectories) {
    if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
        GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
    }

    TempDir disk_dir;
    auto make_resource = [&] {
        return BufferResource::create(
            rmm::mr::get_current_device_resource_ref(),
            PinnedMemoryDisabled,
            {},
            std::nullopt,
            std::make_shared<StreamPool>(1),
            Statistics::disabled(),
            disk_dir.path()
        );
    };
    auto first = make_resource();
    auto second = make_resource();
    auto const first_dir = first->disk_resource()->directory();
    auto const second_dir = second->disk_resource()->directory();

    EXPECT_NE(first_dir, second_dir);
    EXPECT_EQ(first_dir.parent_path(), second_dir.parent_path());
    EXPECT_EQ(first_dir.parent_path(), disk_dir.path());

    first.reset();
    EXPECT_FALSE(std::filesystem::exists(first_dir));
    EXPECT_TRUE(std::filesystem::is_directory(second_dir));
}

TEST_F(DiskResourceTest, DiskBufferDestructorRemovesFile) {
    {
        DiskBuffer disk_buffer{disk_, 0, stream_};
        EXPECT_EQ(files_in(disk_->directory()).size(), 1);
    }
    EXPECT_TRUE(files_in(disk_->directory()).empty());
}

TEST_F(DiskResourceTest, DiskBufferMoveAssignmentReplacesBackingFile) {
    auto const target_data = make_pattern(1024);
    auto const source_data = make_pattern(2048);
    auto target = make_disk_buffer(disk_, target_data);
    {
        auto source = make_disk_buffer(disk_, source_data);
        ASSERT_EQ(files_in(disk_->directory()).size(), 2);

        target = std::move(source);

        // The target's old file is deleted and it now holds the source's.
        EXPECT_EQ(files_in(disk_->directory()).size(), 1);
        EXPECT_TRUE(bytes_equal(target.copy_to_uint8_vector(), source_data));
        EXPECT_EQ(source.size(), 0);  // NOLINT(bugprone-use-after-move)
        EXPECT_THROW(
            std::ignore = source.file_size(),  // NOLINT(bugprone-use-after-move)
            std::logic_error
        );
    }
    // The moved-from source must not have deleted the file it gave away.
    EXPECT_EQ(files_in(disk_->directory()).size(), 1);
    EXPECT_TRUE(bytes_equal(target.copy_to_uint8_vector(), source_data));

    // Self move-assignment keeps the file.
    auto& alias = target;
    target = std::move(alias);
    EXPECT_EQ(files_in(disk_->directory()).size(), 1);
    EXPECT_TRUE(bytes_equal(target.copy_to_uint8_vector(), source_data));
}

TEST_F(DiskResourceTest, DiskBufferMoveAssignmentTakesSourceDiskResource) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();
    ASSERT_NE(other_disk->directory(), disk_->directory());

    auto const pattern = make_pattern(1234);
    DiskBuffer target{disk_, 0, stream_};
    auto source = make_disk_buffer(other_disk, pattern);

    target = std::move(source);

    // The file stays where it was and is now owned by the target.
    EXPECT_EQ(target.disk_resource(), other_disk);
    EXPECT_EQ(files_in(other_disk->directory()).size(), 1);
    EXPECT_TRUE(files_in(disk_->directory()).empty());
    EXPECT_EQ(target.size(), pattern.size());
    EXPECT_TRUE(bytes_equal(target.copy_to_uint8_vector(), pattern));
    EXPECT_EQ(source.size(), 0);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DiskResourceTest, DiskBufferRelocateMovesFileToOtherDiskResource) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();
    ASSERT_NE(other_disk->directory(), disk_->directory());

    auto const pattern = make_pattern(4 * 1024);
    auto buffer = make_disk_buffer(disk_, pattern);

    buffer.relocate(other_disk);

    EXPECT_EQ(buffer.disk_resource(), other_disk);
    EXPECT_TRUE(files_in(disk_->directory()).empty());
    EXPECT_EQ(files_in(other_disk->directory()).size(), 1);
    EXPECT_EQ(buffer.size(), pattern.size());
    EXPECT_TRUE(bytes_equal(buffer.copy_to_uint8_vector(), pattern));

    // Relocating to the current resource keeps the file where it is.
    auto const files = files_in(other_disk->directory());
    buffer.relocate(other_disk);
    EXPECT_EQ(files_in(other_disk->directory()), files);
    EXPECT_TRUE(bytes_equal(buffer.copy_to_uint8_vector(), pattern));
}

TEST_F(DiskResourceTest, DiskBufferRelocateFailureLeavesBufferUnchanged) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();

    DiskBuffer buffer{disk_, 0, stream_};
    EXPECT_THROW(buffer.relocate(nullptr), std::logic_error);

    // Remove the backing file behind the buffer's back so adopt() rejects it.
    auto const files = files_in(disk_->directory());
    ASSERT_EQ(files.size(), 1);
    ASSERT_TRUE(std::filesystem::remove(files.front()));
    EXPECT_THROW(buffer.relocate(other_disk), std::invalid_argument);
    EXPECT_EQ(buffer.disk_resource(), disk_);
    EXPECT_TRUE(files_in(other_disk->directory()).empty());
}

TEST_F(DiskResourceTest, AdoptKeepsOwnedFileAndRelocatesForeignFile) {
    auto const pattern = make_pattern(1024);

    // A file already under the directory is returned unchanged.
    auto const own_path = disk_->create_unique_path();
    EXPECT_EQ(disk_->adopt(own_path), own_path);
    EXPECT_TRUE(std::filesystem::exists(own_path));
    ASSERT_TRUE(std::filesystem::remove(own_path));

    // A file elsewhere is moved under the directory.
    auto const foreign_path = test_path("adopt-foreign");
    EXPECT_EQ(
        disk_->write(foreign_path, pattern.data(), pattern.size(), stream_),
        pattern.size()
    );
    auto const adopted = disk_->adopt(foreign_path);
    EXPECT_EQ(adopted.parent_path(), disk_->directory());
    EXPECT_FALSE(std::filesystem::exists(foreign_path));
    check_file_contents(adopted, 0, pattern);
    ASSERT_TRUE(std::filesystem::remove(adopted));

    // A missing file or a non-regular file is rejected and leaves nothing behind.
    EXPECT_THROW(
        std::ignore = disk_->adopt(test_path("adopt-missing")), std::invalid_argument
    );
    EXPECT_THROW(std::ignore = disk_->adopt(disk_->directory()), std::invalid_argument);
    EXPECT_TRUE(std::filesystem::is_empty(disk_->directory()));

    // A different spelling of a path already under the directory is not renamed.
    auto const own_path2 = disk_->create_unique_path();
    auto const respelled = disk_->directory() / "." / own_path2.filename();
    EXPECT_EQ(disk_->adopt(respelled), respelled);
    EXPECT_TRUE(std::filesystem::exists(own_path2));
    ASSERT_TRUE(std::filesystem::remove(own_path2));
}

TEST_F(DiskResourceTest, DiskBufferReleaseHandsFileToAnotherBuffer) {
    auto const pattern = make_pattern(4 * 1024);

    // Release the file from one buffer.
    std::filesystem::path released;
    {
        auto first = make_disk_buffer(disk_, pattern);
        released = first.release();
        EXPECT_EQ(first.size(), 0);
        EXPECT_TRUE(first.release().empty());
        // Without a backing file, querying the file is a usage error.
        EXPECT_THROW(std::ignore = first.file_size(), std::logic_error);
        EXPECT_TRUE(first.is_empty());  // Zero size, so the file is not queried.
        EXPECT_THROW(std::ignore = first.copy_to_uint8_vector(), std::logic_error);
    }
    // Destruction must not delete a released file; the caller owns it now.
    ASSERT_TRUE(std::filesystem::exists(released));

    // Hand it to another buffer, which infers the size and takes ownership.
    {
        DiskBuffer second{disk_, released, stream_};
        EXPECT_EQ(second.size(), pattern.size());
        EXPECT_EQ(files_in(disk_->directory()).size(), 1);
        EXPECT_TRUE(bytes_equal(second.copy_to_uint8_vector(), pattern));
    }
    // The second buffer owns the adopted file and deletes it.
    EXPECT_FALSE(std::filesystem::exists(released));

    // Invalid arguments.
    EXPECT_THROW(
        DiskBuffer(disk_, test_path("buffer-missing"), stream_), std::invalid_argument
    );
    EXPECT_THROW(DiskBuffer(nullptr, released, stream_), std::logic_error);
}

TEST_F(DiskResourceTest, DiskBufferZeroSizeIsEmpty) {
    DiskBuffer disk_buffer{disk_, 0, stream_};
    EXPECT_TRUE(disk_buffer.is_empty());
}

TEST_F(DiskResourceTest, DiskBufferTracksStream) {
    DiskBuffer disk_buffer{disk_, 0, stream_};
    EXPECT_EQ(disk_buffer.stream().get(), stream_.get());

    auto const new_stream = cuda::stream_ref{cudaStreamPerThread};
    disk_buffer.set_stream(new_stream);
    EXPECT_EQ(disk_buffer.stream().get(), new_stream.get());
}

TEST_F(DiskResourceTest, DiskBufferReportsFileSize) {
    auto const pattern = make_pattern(1024);
    DiskBuffer disk_buffer{disk_, pattern.size(), stream_};
    EXPECT_EQ(disk_buffer.size(), pattern.size());
    EXPECT_EQ(disk_buffer.file_size(), 0U);
    EXPECT_TRUE(disk_buffer.is_empty());
    EXPECT_TRUE(disk_buffer.copy_to_uint8_vector().empty());

    // Write to the buffer's file behind its back, to exercise the size checks.
    auto const files = files_in(disk_->directory());
    ASSERT_EQ(files.size(), 1);
    auto const& file = files.front();
    EXPECT_EQ(
        disk_->write(file, pattern.data(), pattern.size(), stream_), pattern.size()
    );

    EXPECT_EQ(disk_buffer.file_size(), pattern.size());
    EXPECT_FALSE(disk_buffer.is_empty());
    EXPECT_TRUE(bytes_equal(disk_buffer.copy_to_uint8_vector(), pattern));

    // A file larger than the logical size is rejected instead of being read.
    auto const extra = make_pattern(pattern.size() + 1);
    EXPECT_EQ(disk_->write(file, extra.data(), extra.size(), stream_), extra.size());
    EXPECT_THROW(std::ignore = disk_buffer.copy_to_uint8_vector(), std::runtime_error);
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
    EXPECT_EQ(br->disk_resource()->directory().parent_path(), disk_dir.path());
    EXPECT_TRUE(br->is_reservable(MemoryType::DISK));
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
