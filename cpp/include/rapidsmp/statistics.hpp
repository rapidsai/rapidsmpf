/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/communicator/communicator.hpp>

namespace rapidsmp {


/**
 * @brief Track statistics across RAPIDSMP operations.
 */
class Statistics {
  public:
    /**
     * @brief Stores statistics for a single peer.
     */
    struct PeerStats {
        std::size_t comm_count{0};  ///< Number of messages communicated.
        std::size_t comm_nbytes{0};  ///< Number of bytes communicated.

        /**
         * @brief Compares two PeerStats objects for equality.
         * @param o The other PeerStats object to compare with.
         * @return Answer.
         */
        bool operator==(PeerStats const& o) const noexcept {
            return comm_count == o.comm_count && comm_nbytes == o.comm_nbytes;
        }
    };

    /**
     * @brief Constructs a statistics object with a specified number of ranks (peers).
     *
     * @param nranks The number of ranks in the world.
     */
    Statistics(Rank nranks = 0) : nranks_{nranks} {
        peer_stats_.resize(nranks);
    }

    ~Statistics() noexcept = default;

    Statistics(const Statistics&) = delete;
    Statistics& operator=(const Statistics&) = delete;

    /**
     * @brief Move constructor.
     * @param o The Statistics object to move from.
     */
    Statistics(Statistics&& o) noexcept
        : nranks_(o.nranks_), peer_stats_{std::move(o.peer_stats_)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The Statistics object to move from.
     * @return A reference to the updated Statistics object.
     */
    Statistics& operator=(Statistics&& o) noexcept {
        nranks_ = o.nranks_;
        peer_stats_ = std::move(o.peer_stats_);
        return *this;
    }

    /**
     * @brief Checks if the Statistics object is enabled (i.e., has at least one rank).
     * @return True if the object is enabled, otherwise false.
     */
    bool enabled() const noexcept {
        return nranks_ > 0;
    }

    /**
     * @brief Retrieves the statistics for a given peer.
     *
     * @param peer The rank of the peer to retrieve statistics for.
     * @return A PeerStats object for the specified peer.
     */
    PeerStats get_peer_stats(Rank peer) const {
        if (!enabled()) {
            return PeerStats{};
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_stats_.at(peer);
    }

    /**
     * @brief Add peer communication to the statistics.
     *
     * @param peer The rank of the peer.
     * @param nbytes The number of bytes communicated.
     * @return The total number of bytes communicated to the peer after the update.
     */
    std::size_t add_peer_comm(Rank peer, std::size_t nbytes) {
        if (!enabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto& p = peer_stats_.at(peer);
        ++p.comm_count;
        return p.comm_nbytes += nbytes;
    }

    /**
     * @brief Generates a report of statistics in a formatted string.
     *
     * @param column_width The width of each column in the report.
     * @param label_width The width of the labels in the report.
     * @return A string representing the formatted statistics report.
     */
    std::string report(int column_width = 12, int label_width = 22) const {
        if (!enabled()) {
            return "Statistics: disabled";
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::stringstream ss;
        ss << "Statistics:\n";
        ss << std::setw(label_width - 3) << std::left << " - peers:";
        for (Rank i = 0; i < nranks_; ++i) {
            ss << std::right << std::setw(column_width) << "Rank" << i;
        }
        ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-total:";
        for (Rank i = 0; i < nranks_; ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(peer_stats_.at(i).comm_nbytes) << " ";
        }
        ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-mean:";
        for (Rank i = 0; i < nranks_; ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(
                      peer_stats_.at(i).comm_nbytes / (double)peer_stats_.at(i).comm_count
                  )
               << " ";
        }
        ss << "\n";
        return ss.str();
    }

  private:
    mutable std::mutex mutex_;
    Rank nranks_;
    std::vector<PeerStats> peer_stats_;
};

}  // namespace rapidsmp
