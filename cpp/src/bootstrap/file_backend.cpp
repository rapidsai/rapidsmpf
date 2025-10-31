/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace bootstrap {

namespace detail {

namespace {}  // namespace

FileBackend::FileBackend(Context ctx) : ctx_{std::move(ctx)} {
    RAPIDSMPF_EXPECTS(
        ctx_.coord_dir.has_value(), "FileBackend requires coord_dir in context"
    );

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
    std::string alive_path = get_rank_alive_path(ctx_.rank);
    write_file(alive_path, std::to_string(getpid()));

    // Note: Do not block in the constructor. Ranks only create their alive file
    // and continue. Synchronization occurs where needed (e.g., get/put/barrier).
}

FileBackend::~FileBackend() {
    // Clean up rank alive file
    try {
        std::error_code ec;
        std::filesystem::remove(get_rank_alive_path(ctx_.rank), ec);
    } catch (...) {
        // Ignore cleanup errors
    }

    // Rank 0 cleans up the coordination directory when all ranks are done
    if (ctx_.rank == 0) {
        // Note: In a more robust implementation, we might want to wait for
        // all ranks to finish before cleanup, but for simplicity we skip that here.
        // The OS will clean up /tmp directories anyway.
    }
}

void FileBackend::put(std::string const& key, std::string const& value) {
    std::string path = get_kv_path(key);
    write_file(path, value);
}

std::string FileBackend::get(std::string const& key, std::chrono::milliseconds timeout) {
    std::string path = get_kv_path(key);

    if (!wait_for_file(path, timeout)) {
        throw std::runtime_error(
            "Key '" + key + "' not available within " + std::to_string(timeout.count())
            + "ms timeout"
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
        if (r == ctx_.rank)
            continue;

        std::string other_barrier_file =
            get_barrier_path(barrier_id) + "." + std::to_string(r);
        if (!wait_for_file(other_barrier_file, std::chrono::milliseconds{60000})) {
            throw std::runtime_error(
                "Barrier timeout: rank " + std::to_string(r) + " did not arrive"
            );
        }
    }

    // Clean up our barrier file
    {
        std::error_code ec;
        std::filesystem::remove(my_barrier_file, ec);
    }
}

void FileBackend::broadcast(void* data, std::size_t size, Rank root) {
    if (ctx_.rank == root) {
        // Root writes data
        std::string bcast_data{static_cast<char const*>(data), size};
        put("broadcast_" + std::to_string(root), bcast_data);
    } else {
        // Non-root reads data
        std::string bcast_data =
            get("broadcast_" + std::to_string(root), std::chrono::milliseconds{30000});
        if (bcast_data.size() != size) {
            throw std::runtime_error(
                "Broadcast size mismatch: expected " + std::to_string(size) + ", got "
                + std::to_string(bcast_data.size())
            );
        }
        std::memcpy(data, bcast_data.data(), size);
    }

    barrier();
}

std::string FileBackend::get_kv_path(std::string const& key) const {
    return kv_dir_ + "/" + key;
}

std::string FileBackend::get_barrier_path(std::size_t barrier_id) const {
    return barrier_dir_ + "/barrier_" + std::to_string(barrier_id);
}

std::string FileBackend::get_rank_alive_path(Rank rank) const {
    return coord_dir_ + "/rank_" + std::to_string(rank) + "_alive";
}

bool FileBackend::wait_for_file(
    std::string const& path, std::chrono::milliseconds timeout
) {
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
            (void)std::filesystem::status(parent_dir, ec);

            auto now = std::chrono::steady_clock::now();
            if (now - last_dir_scan >= std::chrono::milliseconds{500}) {
                std::filesystem::directory_iterator it(
                    parent_dir,
                    std::filesystem::directory_options::skip_permission_denied,
                    ec
                );
                if (!ec) {
                    for (; it != std::filesystem::directory_iterator(); ++it) {
                        (void)it->path();  // no-op; traversal nudges directory cache
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

void FileBackend::write_file(std::string const& path, std::string const& content) {
    std::string tmp_path = path + ".tmp." + std::to_string(getpid());

    // Write to temporary file
    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open temporary file: " + tmp_path);
    }
    ofs << content;
    ofs.close();

    // Atomic rename
    {
        std::error_code ec;
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) {
            std::error_code rm_ec;
            std::filesystem::remove(tmp_path, rm_ec);  // Clean up temp file
            throw std::runtime_error(
                "Failed to rename " + tmp_path + " to " + path + ": " + ec.message()
            );
        }
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

}  // namespace detail

}  // namespace bootstrap

}  // namespace rapidsmpf
