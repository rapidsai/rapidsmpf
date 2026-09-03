/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/disk/disk_buffer.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/string.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

namespace {

using namespace disk;

std::string test_path(std::string_view suffix) {
    static std::atomic<std::uint64_t> counter{0};
    auto const id = counter.fetch_add(1, std::memory_order_relaxed);
    return (default_spill_directory(GlobalEnvironment->options())
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

void fill_buffer(
    Buffer& buffer, std::vector<std::byte> const& pattern, std::size_t offset
) {
    buffer.write_access([&](std::byte* ptr, cuda::stream_ref stream) {
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(ptr + offset, pattern.data(), pattern.size(), stream)
        );
    });
    buffer.stream().sync();
}

// DiskResource is blocking and not stream-ordered; take exclusive access after
// the buffer's latest write has completed.
struct ExclusiveBufferAccess {
    explicit ExclusiveBufferAccess(Buffer& buf) : buffer_(buf) {
        buffer_.stream().sync();
        ptr_ = buffer_.exclusive_data_access();
    }

    ~ExclusiveBufferAccess() {
        buffer_.unlock();
    }

    ExclusiveBufferAccess(ExclusiveBufferAccess const&) = delete;
    ExclusiveBufferAccess& operator=(ExclusiveBufferAccess const&) = delete;
    ExclusiveBufferAccess(ExclusiveBufferAccess&&) = delete;
    ExclusiveBufferAccess& operator=(ExclusiveBufferAccess&&) = delete;

    void write(
        DiskResource& disk,
        std::filesystem::path const& path,
        std::size_t size,
        std::size_t ptr_offset = 0,
        std::size_t file_offset = 0
    ) {
        EXPECT_EQ(
            disk.write(path, ptr_ + ptr_offset, size, buffer_.mem_type(), file_offset)
                ->get(),
            size
        );
    }

    void read(
        DiskResource& disk,
        std::filesystem::path const& path,
        std::size_t size,
        std::size_t ptr_offset = 0,
        std::size_t file_offset = 0
    ) {
        EXPECT_EQ(
            disk.read(path, ptr_ + ptr_offset, size, buffer_.mem_type(), file_offset)
                ->get(),
            size
        );
    }

  private:
    Buffer& buffer_;
    std::byte* ptr_{nullptr};
};

std::vector<std::byte> copy_from_buffer(
    Buffer const& buffer, std::size_t size, std::size_t offset
) {
    std::vector<std::byte> result(size);
    RAPIDSMPF_CUDA_TRY(
        cuda_memcpy_async(result.data(), buffer.data() + offset, size, buffer.stream())
    );
    buffer.stream().sync();
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

class DiskResourceTest : public ::testing::TestWithParam<MemoryType> {
  protected:
    void SetUp() override {
        if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
            GTEST_SKIP() << "Disk I/O tests run only in the single-process environment";
        }
        if (GetParam() == MemoryType::PINNED_HOST
            && !is_pinned_memory_resources_supported())
        {
            GTEST_SKIP() << "Pinned memory resources are not supported on this system";
        }

        auto pinned_pool_properties = is_pinned_memory_resources_supported()
                                          ? PinnedPoolProperties{}
                                          : PinnedMemoryDisabled;
        br_ = BufferResource::create(
            rmm::mr::get_current_device_resource_ref(), std::move(pinned_pool_properties)
        );
        disk_ = br_->disk_resource();
        stream_ = cuda::stream_ref{cudaStreamLegacy};
    }

    std::unique_ptr<Buffer> make_buffer(std::size_t size) {
        return br_->make_buffer(stream_, br_->reserve_or_fail(size, GetParam()));
    }

    std::shared_ptr<DiskResource> disk_;
    std::shared_ptr<BufferResource> br_;
    cuda::stream_ref stream_{cudaStreamLegacy};
};

INSTANTIATE_TEST_SUITE_P(
    MemoryTypes,
    DiskResourceTest,
    ::testing::ValuesIn(MEMORY_TYPES),
    [](::testing::TestParamInfo<MemoryType> const& info) { return to_string(info.param); }
);

TEST_P(DiskResourceTest, RoundTrip) {
    auto const path = test_path("roundtrip");
    auto const pattern = make_pattern(64 * 1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);

    ExclusiveBufferAccess{*source}.write(*disk_, path, pattern.size());

    auto destination = make_buffer(pattern.size());
    ExclusiveBufferAccess{*destination}.read(*disk_, path, pattern.size());

    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), 0), pattern);
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_P(DiskResourceTest, UnalignedOffsetRoundTrip) {
    auto const path = test_path("unaligned");
    auto const pattern = make_pattern(16 * 1024);
    auto const ptr_offset = std::size_t{1};
    auto const file_offset = std::size_t{1};

    auto source = make_buffer(pattern.size() + ptr_offset);
    fill_buffer(*source, pattern, ptr_offset);

    ExclusiveBufferAccess{*source}.write(
        *disk_, path, pattern.size(), ptr_offset, file_offset
    );

    EXPECT_EQ(std::filesystem::file_size(path), pattern.size() + file_offset);
    check_file_contents(path, file_offset, pattern);

    auto destination = make_buffer(pattern.size() + ptr_offset);
    ExclusiveBufferAccess{*destination}.read(
        *disk_, path, pattern.size(), ptr_offset, file_offset
    );
    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), ptr_offset), pattern);

    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_P(DiskResourceTest, DiskFutureIsReadyBeforeGet) {
    auto const path = test_path("is-ready");
    auto const pattern = make_pattern(64 * 1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);
    source->stream().sync();

    auto future = disk_->write(path, source->data(), pattern.size(), source->mem_type());
    ASSERT_TRUE(future->valid());
    while (!future->is_ready()) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(future->is_ready());
    EXPECT_EQ(future->get(), pattern.size());
    EXPECT_FALSE(future->valid());
    EXPECT_FALSE(future->is_ready());
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST_P(DiskResourceTest, FlushDoesNotThrow) {
    auto const path = test_path("flush");
    auto const pattern = make_pattern(4096);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);
    ExclusiveBufferAccess{*source}.write(*disk_, path, pattern.size());
    EXPECT_NO_THROW(disk_->flush(path));
    ASSERT_TRUE(std::filesystem::remove(path));
}

TEST(DiskSpillDirectory, EmptyOptionUsesTempDir) {
    config::Options options;
    EXPECT_EQ(default_spill_directory(options), std::filesystem::temp_directory_path());
}

TEST(DiskSpillDirectory, UsesConfiguredPath) {
    config::Options options{
        {{"disk_spill_dir", config::OptionValue("/tmp/rapidsmpf-spill")}}
    };
    EXPECT_EQ(
        default_spill_directory(options), std::filesystem::path{"/tmp/rapidsmpf-spill"}
    );
}

TEST(DiskResource, SharedPtrKeepsDiskResourceAlive) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
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

TEST_P(DiskResourceTest, DiskBufferFromBufferRestoreRoundTrip) {
    auto const pattern = make_pattern(64 * 1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);

    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    auto const path = disk_buf->path();
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(disk_buf->size(), pattern.size());
    EXPECT_EQ(path.parent_path(), br_->disk_resource()->directory());

    auto reservation = br_->reserve_or_fail(pattern.size(), GetParam());
    auto destination = DiskBuffer::restore(std::move(disk_buf), reservation, stream_);
    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), 0), pattern);
}

TEST_P(DiskResourceTest, DiskBufferRecordsCopyStats) {
    auto stats = Statistics::create();
    auto pinned_pool_properties = is_pinned_memory_resources_supported()
                                      ? PinnedPoolProperties{}
                                      : PinnedMemoryDisabled;
    br_ = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        std::move(pinned_pool_properties),
        {},
        std::chrono::milliseconds{1},
        std::make_shared<StreamPool>(16),
        stats
    );
    disk_ = br_->disk_resource();

    auto const pattern = make_pattern(64 * 1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);

    auto const mem_name = to_lower(to_string(GetParam()));
    auto const to_disk_bytes = "copy-" + mem_name + "-to-disk-bytes";
    auto const to_disk_time = "copy-" + mem_name + "-to-disk-time";
    auto const from_disk_bytes = "copy-disk-to-" + mem_name + "-bytes";
    auto const from_disk_time = "copy-disk-to-" + mem_name + "-time";

    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    EXPECT_EQ(stats->get_stat(to_disk_bytes).count(), 1);
    EXPECT_EQ(stats->get_stat(to_disk_bytes).value(), pattern.size());
    EXPECT_GE(stats->get_stat(to_disk_time).value(), 0.0);
    EXPECT_THROW(std::ignore = stats->get_stat(from_disk_bytes), std::out_of_range);

    auto reservation = br_->reserve_or_fail(pattern.size(), GetParam());
    auto destination = DiskBuffer::restore(std::move(disk_buf), reservation, stream_);
    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), 0), pattern);
    EXPECT_EQ(stats->get_stat(from_disk_bytes).count(), 1);
    EXPECT_EQ(stats->get_stat(from_disk_bytes).value(), pattern.size());
    EXPECT_GE(stats->get_stat(from_disk_time).value(), 0.0);
    EXPECT_THAT(stats->report(), ::testing::HasSubstr("copy-" + mem_name + "-to-disk"));
    EXPECT_THAT(stats->report(), ::testing::HasSubstr("copy-disk-to-" + mem_name));
}

TEST_P(DiskResourceTest, DiskBufferZeroSizeRoundTrip) {
    auto source = make_buffer(0);
    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    EXPECT_EQ(disk_buf->size(), 0U);
    EXPECT_TRUE(std::filesystem::exists(disk_buf->path()));

    auto reservation = br_->reserve_or_fail(0, GetParam());
    auto destination = DiskBuffer::restore(std::move(disk_buf), reservation, stream_);
    EXPECT_EQ(destination->size, 0U);
}

TEST_P(DiskResourceTest, DiskBufferMoveTransfersFileOwnership) {
    auto const pattern = make_pattern(4096);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);

    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    auto const path = disk_buf->path();
    auto moved = std::move(disk_buf);
    EXPECT_EQ(disk_buf, nullptr);
    EXPECT_EQ(moved->path(), path);
    EXPECT_TRUE(std::filesystem::exists(path));

    auto reservation = br_->reserve_or_fail(pattern.size(), GetParam());
    auto destination = DiskBuffer::restore(std::move(moved), reservation, stream_);
    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), 0), pattern);
}

TEST_P(DiskResourceTest, DiskBufferDeallocateRemovesFile) {
    auto const pattern = make_pattern(1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);
    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    auto const path = disk_buf->path();
    ASSERT_TRUE(std::filesystem::exists(path));

    disk_buf->deallocate();
    EXPECT_TRUE(disk_buf->path().empty());
    EXPECT_EQ(disk_buf->size(), 0U);
    EXPECT_FALSE(std::filesystem::exists(path));

    EXPECT_NO_THROW(disk_buf->deallocate());
}

TEST_P(DiskResourceTest, DiskBufferDestructorRemovesFile) {
    std::filesystem::path path;
    {
        auto const pattern = make_pattern(1024);
        auto source = make_buffer(pattern.size());
        fill_buffer(*source, pattern, 0);
        auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
        path = disk_buf->path();
        ASSERT_TRUE(std::filesystem::exists(path));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_P(DiskResourceTest, DiskBufferOutlivesBufferResource) {
    auto const pattern = make_pattern(2048);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);

    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);
    auto const path = disk_buf->path();
    br_.reset();

    ASSERT_TRUE(std::filesystem::exists(path));

    auto pinned_pool_properties = is_pinned_memory_resources_supported()
                                      ? PinnedPoolProperties{}
                                      : PinnedMemoryDisabled;
    auto br2 = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), std::move(pinned_pool_properties)
    );
    auto reservation = br2->reserve_or_fail(pattern.size(), GetParam());
    auto destination = DiskBuffer::restore(std::move(disk_buf), reservation, stream_);
    EXPECT_EQ(copy_from_buffer(*destination, pattern.size(), 0), pattern);
}

TEST_P(DiskResourceTest, DiskBufferRestoreRejectsUndersizedReservation) {
    auto const pattern = make_pattern(1024);
    auto source = make_buffer(pattern.size());
    fill_buffer(*source, pattern, 0);
    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br_);

    auto reservation = br_->reserve_or_fail(512, GetParam());
    EXPECT_THROW(
        [&] {
            std::ignore = DiskBuffer::restore(std::move(disk_buf), reservation, stream_);
        }(),
        rapidsmpf::reservation_error
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

    auto const pattern = make_pattern(256);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    auto source =
        br->make_buffer(stream, br->reserve_or_fail(pattern.size(), MemoryType::HOST));
    fill_buffer(*source, pattern, 0);

    auto disk_buf = DiskBuffer::from_buffer(std::move(source), *br);
    EXPECT_EQ(
        disk_buf->path().parent_path(), disk_dir.path() / std::to_string(::getpid())
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

}  // namespace
