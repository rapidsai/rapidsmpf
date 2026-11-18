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

/// Write @p data to a plain file outside any disk resource.
void write_external_file(
    std::filesystem::path const& path, std::vector<std::byte> const& data
) {
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<char const*>(data.data()), std::streamsize(data.size()));
    ASSERT_TRUE(file.good());
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

    /// Check that @p expected is stored at @p file_offset in @p path.
    void check_file_contents(
        FileHandle const& path,
        std::ptrdiff_t file_offset,
        std::vector<std::byte> const& expected
    ) {
        std::vector<std::byte> actual(expected.size());
        ASSERT_EQ(
            disk_->read(path, actual.data(), actual.size(), stream_, file_offset),
            actual.size()
        );
        EXPECT_EQ(actual, expected);
    }

    /// Regular files directly under @p dir.
    static std::size_t count_files(std::filesystem::path const& dir) {
        return static_cast<std::size_t>(std::distance(
            std::filesystem::directory_iterator{dir},
            std::filesystem::directory_iterator{}
        ));
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
    auto path = disk_->create_unique_path();
    auto const pattern = make_pattern(64 * 1024);

    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), stream_), pattern.size()
    );

    std::vector<std::byte> destination(pattern.size());
    EXPECT_EQ(
        disk_->read(path, destination.data(), pattern.size(), stream_), pattern.size()
    );

    EXPECT_EQ(destination, pattern);
    ASSERT_TRUE(disk_->remove(path));
}

TEST_F(DiskResourceTest, UnalignedOffsetRoundTrip) {
    auto path = disk_->create_unique_path();
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

    EXPECT_EQ(path.file_size(), pattern.size() + file_offset);
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

    ASSERT_TRUE(disk_->remove(path));
}

TEST_F(DiskResourceTest, SequentialOffsetWritesPreserveExistingBytes) {
    auto path = disk_->create_unique_path();
    auto const first = make_pattern(8 * 1024);
    auto const second = make_pattern(4 * 1024);
    auto const second_offset = std::ptrdiff_t{12 * 1024};

    EXPECT_EQ(disk_->write(path, first.data(), first.size(), stream_), first.size());
    EXPECT_EQ(
        disk_->write(path, second.data(), second.size(), stream_, second_offset),
        second.size()
    );

    EXPECT_EQ(path.file_size(), second_offset + second.size());
    check_file_contents(path, 0, first);
    check_file_contents(path, second_offset, second);

    ASSERT_TRUE(disk_->remove(path));
}

TEST_F(DiskResourceTest, CopyPreservesBytesOutsideWrittenRange) {
    auto src_path = disk_->create_unique_path();
    auto dst_path = disk_->create_unique_path();
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

    EXPECT_EQ(dst_path.file_size(), dst.size());
    auto expected = dst;
    std::copy_n(src.begin() + src_offset, size, expected.begin() + dst_offset);
    check_file_contents(dst_path, 0, expected);

    ASSERT_TRUE(disk_->remove(src_path));
    ASSERT_TRUE(disk_->remove(dst_path));
}

TEST_F(DiskResourceTest, CopyRejectsInvalidArguments) {
    auto src_path = disk_->create_unique_path();
    auto dst_path = disk_->create_unique_path();
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

    ASSERT_TRUE(disk_->remove(src_path));
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
    write_external_file(src_path, src);

    auto adopted = disk_->adopt(src_path);

    EXPECT_FALSE(std::filesystem::exists(src_path));
    EXPECT_EQ(adopted.file_size(), src.size());
    check_file_contents(adopted, 0, src);
    // The adopted file is the only one under the directory, so no temporary is left.
    EXPECT_EQ(count_files(disk_->directory()), 1);

    ASSERT_TRUE(disk_->remove(adopted));
}

TEST_F(DiskResourceTest, FlushDoesNotThrow) {
    auto path = disk_->create_unique_path();
    auto const pattern = make_pattern(4096);
    EXPECT_EQ(
        disk_->write(path, pattern.data(), pattern.size(), stream_), pattern.size()
    );
    EXPECT_NO_THROW(disk_->flush(path));
    ASSERT_TRUE(disk_->remove(path));
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
        EXPECT_EQ(count_files(disk_->directory()), 1);
    }
    EXPECT_EQ(count_files(disk_->directory()), 0);
}

TEST_F(DiskResourceTest, DiskBufferMoveAssignmentReplacesBackingFile) {
    auto const target_data = make_pattern(1024);
    auto const source_data = make_pattern(2048);
    DiskBuffer target{disk_, target_data.size(), stream_};
    EXPECT_EQ(
        disk_->write(target.path(), target_data.data(), target_data.size(), stream_),
        target_data.size()
    );
    {
        DiskBuffer source{disk_, source_data.size(), stream_};
        EXPECT_EQ(
            disk_->write(source.path(), source_data.data(), source_data.size(), stream_),
            source_data.size()
        );
        auto const source_path = source.path();
        ASSERT_EQ(count_files(disk_->directory()), 2);

        target = std::move(source);

        // The target's old file is deleted and it now holds the source's.
        EXPECT_EQ(count_files(disk_->directory()), 1);
        EXPECT_EQ(target.path(), source_path);
        EXPECT_EQ(target.size(), source_data.size());
        EXPECT_EQ(source.size(), 0);  // NOLINT(bugprone-use-after-move)
        EXPECT_TRUE(source.path().empty());  // NOLINT(bugprone-use-after-move)
    }
    // The moved-from source must not have deleted the file it gave away.
    EXPECT_EQ(count_files(disk_->directory()), 1);
    check_file_contents(target.path(), 0, source_data);

    // Self move-assignment keeps the file.
    auto const path = target.path();
    auto& alias = target;
    target = std::move(alias);
    EXPECT_EQ(target.path(), path);
    check_file_contents(target.path(), 0, source_data);
}

TEST_F(DiskResourceTest, DiskBufferMoveAssignmentTakesSourceDiskResource) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();
    ASSERT_NE(other_disk->directory(), disk_->directory());

    auto const pattern = make_pattern(1234);
    DiskBuffer target{disk_, 0, stream_};
    DiskBuffer source{other_disk, pattern.size(), stream_};
    EXPECT_EQ(
        other_disk->write(source.path(), pattern.data(), pattern.size(), stream_),
        pattern.size()
    );
    auto const source_path = source.path();

    target = std::move(source);

    // The file stays where it was and is now owned by the target. The target's old file
    // was deleted through the resource that owned it.
    EXPECT_EQ(target.disk_resource(), other_disk);
    EXPECT_EQ(target.path(), source_path);
    EXPECT_EQ(count_files(other_disk->directory()), 1);
    EXPECT_EQ(count_files(disk_->directory()), 0);
    EXPECT_EQ(target.size(), pattern.size());
    EXPECT_EQ(target.path().file_size(), pattern.size());
    EXPECT_EQ(source.size(), 0);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DiskResourceTest, DiskBufferRelocateMovesFileToOtherDiskResource) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();
    ASSERT_NE(other_disk->directory(), disk_->directory());

    auto const pattern = make_pattern(4 * 1024);
    DiskBuffer buffer{disk_, pattern.size(), stream_};
    EXPECT_EQ(
        disk_->write(buffer.path(), pattern.data(), pattern.size(), stream_),
        pattern.size()
    );

    buffer.relocate(other_disk);

    EXPECT_EQ(buffer.disk_resource(), other_disk);
    EXPECT_EQ(count_files(disk_->directory()), 0);
    EXPECT_EQ(count_files(other_disk->directory()), 1);
    EXPECT_EQ(buffer.size(), pattern.size());
    EXPECT_EQ(buffer.path().file_size(), pattern.size());
    EXPECT_FALSE(buffer.copy_to_uint8_vector().empty());

    // Relocating to the current resource keeps the file where it is.
    auto const path = buffer.path();
    buffer.relocate(other_disk);
    EXPECT_EQ(buffer.path(), path);
    EXPECT_EQ(count_files(other_disk->directory()), 1);
}

TEST_F(DiskResourceTest, DiskBufferRelocateFailureLeavesBufferUnchanged) {
    auto other_br = make_other_br();
    auto other_disk = other_br->disk_resource();

    DiskBuffer buffer{disk_, 0, stream_};
    EXPECT_THROW(buffer.relocate(nullptr), std::logic_error);

    // Remove the backing file behind the buffer's back so adopt() rejects it.
    auto const path = buffer.path();
    auto const entry = std::filesystem::directory_iterator{disk_->directory()}->path();
    ASSERT_TRUE(std::filesystem::remove(entry));
    EXPECT_THROW(buffer.relocate(other_disk), std::invalid_argument);
    EXPECT_EQ(buffer.disk_resource(), disk_);
    EXPECT_EQ(buffer.path(), path);
    EXPECT_EQ(count_files(other_disk->directory()), 0);
}

TEST_F(DiskResourceTest, AdoptKeepsOwnedFileAndRelocatesForeignFile) {
    auto const pattern = make_pattern(1024);

    // A file already under the directory is returned unchanged.
    auto own_path = disk_->create_unique_path();
    EXPECT_EQ(disk_->adopt(own_path), own_path);
    EXPECT_EQ(count_files(disk_->directory()), 1);
    ASSERT_TRUE(disk_->remove(own_path));

    // An external file is moved under the directory.
    TempDir external;
    auto const foreign_path = external.path() / "adopt-foreign";
    write_external_file(foreign_path, pattern);
    auto adopted = disk_->adopt(foreign_path);
    EXPECT_FALSE(std::filesystem::exists(foreign_path));
    EXPECT_EQ(count_files(disk_->directory()), 1);
    check_file_contents(adopted, 0, pattern);
    ASSERT_TRUE(disk_->remove(adopted));

    // A missing file, a directory or an empty handle is rejected and leaves nothing
    // behind.
    EXPECT_THROW(
        std::ignore = disk_->adopt(external.path() / "adopt-missing"),
        std::invalid_argument
    );
    EXPECT_THROW(std::ignore = disk_->adopt(disk_->directory()), std::invalid_argument);
    EXPECT_THROW(std::ignore = disk_->adopt(FileHandle{}), std::invalid_argument);
    EXPECT_EQ(count_files(disk_->directory()), 0);

    // A different spelling of a path already under the directory is not renamed.
    own_path = disk_->create_unique_path();
    auto const entry = std::filesystem::directory_iterator{disk_->directory()}->path();
    auto const respelled = disk_->directory() / "." / entry.filename();
    std::ignore = disk_->adopt(respelled);
    EXPECT_TRUE(std::filesystem::exists(entry));
    EXPECT_EQ(count_files(disk_->directory()), 1);
    ASSERT_TRUE(disk_->remove(own_path));
}

TEST_F(DiskResourceTest, DiskBufferReleaseHandsFileToAnotherBuffer) {
    auto const pattern = make_pattern(4 * 1024);

    // Release the file from one buffer.
    FileHandle released;
    {
        DiskBuffer first{disk_, pattern.size(), stream_};
        EXPECT_EQ(
            disk_->write(first.path(), pattern.data(), pattern.size(), stream_),
            pattern.size()
        );
        released = first.release();
        EXPECT_FALSE(released.empty());
        EXPECT_EQ(first.size(), 0);
        EXPECT_TRUE(first.path().empty());
        EXPECT_TRUE(first.release().empty());
        // Without a backing file, querying the file is a usage error.
        EXPECT_THROW(std::ignore = first.file_size(), std::logic_error);
        EXPECT_TRUE(first.is_empty());  // Zero size, so the file is not queried.
        EXPECT_THROW(std::ignore = first.copy_to_uint8_vector(), std::logic_error);
    }
    // Destruction must not delete a released file; the caller owns it now.
    EXPECT_EQ(count_files(disk_->directory()), 1);

    // Hand it to another buffer, which infers the size and takes ownership.
    {
        DiskBuffer second{disk_, released, stream_};
        EXPECT_EQ(second.path(), released);  // Already under the directory.
        EXPECT_EQ(second.size(), pattern.size());
        check_file_contents(second.path(), 0, pattern);
    }
    // The second buffer owns the adopted file and deletes it.
    EXPECT_EQ(count_files(disk_->directory()), 0);

    // Invalid arguments.
    TempDir external;
    EXPECT_THROW(
        DiskBuffer(disk_, disk_->adopt(external.path() / "buffer-missing"), stream_),
        std::invalid_argument
    );
    EXPECT_THROW(DiskBuffer(disk_, FileHandle{}, stream_), std::invalid_argument);
    EXPECT_THROW(DiskBuffer(nullptr, FileHandle{}, stream_), std::logic_error);
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

    EXPECT_EQ(
        disk_->write(disk_buffer.path(), pattern.data(), pattern.size(), stream_),
        pattern.size()
    );

    EXPECT_EQ(disk_buffer.file_size(), pattern.size());
    EXPECT_FALSE(disk_buffer.is_empty());
    EXPECT_TRUE(
        std::ranges::equal(
            disk_buffer.copy_to_uint8_vector(),
            pattern,
            [](std::uint8_t lhs, std::byte rhs) {
                return lhs == std::to_integer<std::uint8_t>(rhs);
            }
        )
    );

    // A file larger than the logical size is rejected instead of being read.
    auto const extra = make_pattern(pattern.size() + 1);
    EXPECT_EQ(
        disk_->write(disk_buffer.path(), extra.data(), extra.size(), stream_),
        extra.size()
    );
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
