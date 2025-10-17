/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace bootstrap {

namespace detail {

namespace {

/**
 * @brief Create directory recursively (like mkdir -p).
 */
void mkdir_p(std::string const& path) {
    if (path.empty() || path == "/") {
        return;
    }

    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return;  // Directory already exists
        } else {
            throw std::runtime_error("Path exists but is not a directory: " + path);
        }
    }

    // Create parent directory first
    auto pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        mkdir_p(path.substr(0, pos));
    }

    // Create this directory
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error(
            "Failed to create directory " + path + ": " + std::strerror(errno)
        );
    }
}

}  // namespace

FileBackend::FileBackend(Context const& ctx) : ctx_{ctx} {
    RAPIDSMPF_EXPECTS(
        ctx_.coord_dir.has_value(), "FileBackend requires coord_dir in context"
    );

    coord_dir_ = *ctx_.coord_dir;
    kv_dir_ = coord_dir_ + "/kv";
    barrier_dir_ = coord_dir_ + "/barriers";

    try {
        mkdir_p(coord_dir_);
        mkdir_p(kv_dir_);
        mkdir_p(barrier_dir_);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            "Failed to initialize coordination directory structure: "
            + std::string{e.what()}
        );
    }

    // Create rank alive file
    std::string alive_path = get_rank_alive_path(ctx_.rank);
    write_file(alive_path, std::to_string(getpid()));

    // Wait for all ranks to be alive (optional health check)
    // This helps detect early failures
    if (ctx_.rank == 0) {
        // Rank 0 waits for all other ranks
        for (Rank r = 1; r < ctx_.nranks; ++r) {
            if (!wait_for_file(get_rank_alive_path(r), std::chrono::milliseconds{30000}))
            {
                throw std::runtime_error(
                    "Rank " + std::to_string(r) + " did not signal alive within timeout"
                );
            }
        }
        // Signal that initialization is complete
        write_file(coord_dir_ + "/initialized", "1");
    } else {
        // Other ranks wait for rank 0 to complete initialization
        if (!wait_for_file(coord_dir_ + "/initialized", std::chrono::milliseconds{30000}))
        {
            throw std::runtime_error(
                "Rank 0 did not complete initialization within timeout"
            );
        }
    }
}

FileBackend::~FileBackend() {
    // Clean up rank alive file
    try {
        unlink(get_rank_alive_path(ctx_.rank).c_str());
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

std::string FileBackend::get(std::string const& key, int timeout_ms) {
    std::string path = get_kv_path(key);

    if (!wait_for_file(path, std::chrono::milliseconds{timeout_ms})) {
        throw std::runtime_error(
            "Key '" + key + "' not available within " + std::to_string(timeout_ms)
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
    unlink(my_barrier_file.c_str());
}

void FileBackend::broadcast(void* data, std::size_t size, Rank root) {
    if (ctx_.rank == root) {
        // Root writes data
        std::string bcast_data{static_cast<char const*>(data), size};
        put("broadcast_" + std::to_string(root), bcast_data);
    } else {
        // Non-root reads data
        std::string bcast_data = get("broadcast_" + std::to_string(root), 30000);
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

    while (true) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            return true;  // File exists
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            return false;  // Timeout
        }

        // Sleep before next poll
        std::this_thread::sleep_for(poll_interval);

        // Exponential backoff up to 100ms
        if (poll_interval < std::chrono::milliseconds{100}) {
            poll_interval = std::min(poll_interval * 2, std::chrono::milliseconds{100});
        }
    }
}

void FileBackend::ensure_directory(std::string const& path) {
    mkdir_p(path);
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
    if (rename(tmp_path.c_str(), path.c_str()) != 0) {
        unlink(tmp_path.c_str());  // Clean up temp file
        throw std::runtime_error(
            "Failed to rename " + tmp_path + " to " + path + ": " + std::strerror(errno)
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

}  // namespace detail

}  // namespace bootstrap

}  // namespace rapidsmpf
