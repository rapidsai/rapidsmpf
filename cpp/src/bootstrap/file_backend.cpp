/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>

#include <unistd.h>

#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/error.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rapidsmpf::bootstrap::detail {

FileBackend::FileBackend(Context ctx) : ctx_{std::move(ctx)} {
    if (!ctx_.coord_dir.has_value()) {
        throw std::runtime_error("FileBackend requires coord_dir in context");
    }

    coord_dir_ = *ctx_.coord_dir;
    kv_dir_ = coord_dir_ + "/kv";
    barrier_dir_ = coord_dir_ + "/barriers";

    try {
        std::filesystem::create_directories(coord_dir_);
        std::filesystem::create_directories(kv_dir_);
        std::filesystem::create_directories(barrier_dir_);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            "Failed to initialize coordination directory structure: "
            + std::string{e.what()}
        );
    }

    // Create rank alive file
    write_file(get_rank_alive_path(ctx_.rank), std::to_string(getpid()));

    // Note: Do not block in the constructor. Ranks only create their alive file
    // and continue. Synchronization occurs where needed (e.g., get/put/barrier).
}

FileBackend::~FileBackend() {
    // Clean up rank alive file
    try {
        std::error_code ec;
        if (!std::filesystem::remove(get_rank_alive_path(ctx_.rank), ec) && ec) {
            std::cerr << "Error removing rank alive file: " << ec.message() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during rank alive file cleanup: " << e.what()
                  << std::endl;
    }
    cleanup_coordination_directory();
}

void FileBackend::put(std::string const& key, std::string_view value) {
    if (ctx_.rank != 0) {
        throw std::runtime_error(
            "put() can only be called by rank 0, but was called by rank "
            + std::to_string(ctx_.rank)
        );
    }

    std::string path = get_kv_path(key);
    write_file(get_kv_path(key), value);
}

std::string FileBackend::get(std::string const& key, Duration timeout) {
    std::string path = get_kv_path(key);
    auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
    if (!wait_for_file(path, timeout_ms)) {
        throw std::runtime_error(
            "Key '" + key + "' not available within " + std::to_string(timeout.count())
            + "s timeout"
        );
    }
    return read_file(path);
}

void FileBackend::barrier() {
    std::size_t barrier_id = barrier_count_++;
    std::string my_barrier_file =
        get_barrier_path(barrier_id) + "." + std::to_string(ctx_.rank);

    // Each rank creates its barrier file
    write_file(my_barrier_file, "1");

    // Wait for all other ranks
    for (Rank r = 0; r < ctx_.nranks; ++r) {
        if (r == ctx_.rank) {
            continue;
        }

        std::string other_barrier_file =
            get_barrier_path(barrier_id) + "." + std::to_string(r);
        if (!wait_for_file(other_barrier_file, std::chrono::milliseconds{60000})) {
            throw std::runtime_error(
                "Barrier timeout: rank " + std::to_string(r) + " did not arrive"
            );
        }
    }

    // Clean up our barrier file
    std::error_code ec;
    std::filesystem::remove(my_barrier_file, ec);
}

void FileBackend::sync() {
    // For FileBackend, this is a no-op since put() operations use atomic
    // file writes that are immediately visible to all processes via the
    // shared filesystem.
}

std::string FileBackend::get_kv_path(std::string const& key) const {
    validate_key(key);
    return kv_dir_ + "/" + key;
}

std::string FileBackend::get_barrier_path(std::size_t barrier_id) const {
    return barrier_dir_ + "/barrier_" + std::to_string(barrier_id);
}

std::string FileBackend::get_rank_alive_path(Rank rank) const {
    return coord_dir_ + "/rank_" + std::to_string(rank) + "_alive";
}

bool FileBackend::wait_for_file(std::string const& path, Duration timeout) {
    auto start = std::chrono::steady_clock::now();
    auto poll_interval = std::chrono::milliseconds{10};

    // NFS visibility aid: derive parent directory to refresh its metadata
    std::string parent_dir;
    {
        auto pos = path.find_last_of('/');
        if (pos != std::string::npos && pos > 0) {
            parent_dir = path.substr(0, pos);
        }
    }
    auto last_dir_scan = start - std::chrono::milliseconds{1000};

    while (true) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            return true;  // File exists
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            return false;  // Timeout
        }

        // Hint NFS to refresh directory cache: status and occasionally iterate directory
        // on parent. Without this remote processes may timeout to spawn because NFS never
        // refreshes.
        if (!parent_dir.empty()) {
            std::ignore = std::filesystem::status(parent_dir, ec);

            auto now = std::chrono::steady_clock::now();
            if (now - last_dir_scan >= std::chrono::milliseconds{500}) {
                std::filesystem::directory_iterator it(
                    parent_dir,
                    std::filesystem::directory_options::skip_permission_denied,
                    ec
                );
                if (!ec) {
                    for (; it != std::filesystem::directory_iterator(); ++it) {
                        std::ignore =
                            it->path();  // no-op; traversal nudges directory cache
                    }
                }
                last_dir_scan = now;
            }
        }

        // Sleep before next poll
        std::this_thread::sleep_for(poll_interval);

        // Exponential backoff up to 100ms
        if (poll_interval < std::chrono::milliseconds{100}) {
            poll_interval = std::min(poll_interval * 2, std::chrono::milliseconds{100});
        }
    }
}

void FileBackend::write_file(std::string const& path, std::string_view content) {
    // Create the temp file in the parent directory rather than appending to the key
    // name, so a max-length (255-byte) key does not push the filename past NAME_MAX.
    auto slash = path.rfind('/');
    std::string parent = (slash != std::string::npos) ? path.substr(0, slash) : ".";
    std::string tmp_path = parent + "/.tmp.XXXXXX";

    // mkstemp requires a mutable char array and atomically creates a unique file,
    // preventing symlink race conditions on shared filesystems.
    int fd = mkstemp(tmp_path.data());
    if (fd == -1) {
        throw std::runtime_error(
            "Failed to create temporary file via mkstemp: " + tmp_path + ": "
            + std::strerror(errno)
        );
    }

    // Write content and close the file descriptor
    auto bytes_left = content.size();
    auto const* ptr = content.data();
    while (bytes_left > 0) {
        auto written = ::write(fd, ptr, bytes_left);
        if (written < 0) {
            if (errno == EINTR) {
                continue;  // re-try on interrupt
            }
            int err = errno;
            ::close(fd);
            ::unlink(tmp_path.c_str());
            throw std::runtime_error(
                "Failed to write to temporary file: " + tmp_path + ": "
                + std::strerror(err)
            );
        }
        bytes_left -= static_cast<std::size_t>(written);
        ptr += written;
    }
    ::close(fd);

    // POSIX rename(2) guarantees atomic replacement of the destination.
    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        int err = errno;
        ::unlink(tmp_path.c_str());
        throw std::runtime_error(
            "Failed to rename " + tmp_path + " to " + path + ": " + std::strerror(err)
        );
    }
}

std::string FileBackend::read_file(std::string const& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

void FileBackend::cleanup_coordination_directory() {
    // Only rank 0 performs cleanup; other ranks return immediately
    if (ctx_.rank != 0) {
        return;
    }

    // Wait for all other ranks to clean up their alive files
    auto cleanup_timeout = std::chrono::seconds{30};
    auto start = std::chrono::steady_clock::now();
    auto poll_interval = std::chrono::milliseconds{100};

    bool all_ranks_done = false;
    while (!all_ranks_done) {
        all_ranks_done = true;

        // Check if all other ranks' alive files are gone
        for (Rank r = 0; r < ctx_.nranks; ++r) {
            if (r == ctx_.rank) {
                continue;
            }

            std::error_code ec;
            std::string alive_path = get_rank_alive_path(r);
            if (std::filesystem::exists(alive_path, ec)) {
                all_ranks_done = false;
                break;
            }
        }

        if (all_ranks_done) {
            break;
        }

        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= cleanup_timeout) {
            std::cerr << "Warning: Timeout waiting for all ranks to finish. "
                      << "Some alive files may still exist. Proceeding with cleanup."
                      << std::endl;
            break;
        }

        // Sleep before next poll
        std::this_thread::sleep_for(poll_interval);
    }

    // Clean up the entire coordination directory
    try {
        std::error_code ec;
        if (std::filesystem::remove_all(coord_dir_, ec) == 0 && ec) {
            std::cerr << "Warning: Failed to remove coordination directory '"
                      << coord_dir_ << "': " << ec.message() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during coordination directory cleanup: " << e.what()
                  << std::endl;
    }
}
}  // namespace rapidsmpf::bootstrap::detail
