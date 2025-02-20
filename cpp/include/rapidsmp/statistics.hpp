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


class Statistics {
  public:
    struct PeerStats {
        std::size_t send_count{0};
        std::size_t send_nbytes{0};

        bool operator==(PeerStats const& o) const noexcept {
            return send_count == o.send_count && send_nbytes == o.send_nbytes;
        }
    };

    Statistics(Rank nranks = 0) : nranks_{nranks} {
        peer_stats_.resize(nranks);
    }

    ~Statistics() noexcept = default;

    Statistics(const Statistics&) = delete;
    Statistics& operator=(const Statistics&) = delete;

    Statistics(Statistics&& o) noexcept
        : nranks_(o.nranks_), peer_stats_{std::move(o.peer_stats_)} {}

    Statistics& operator=(Statistics&& o) noexcept {
        nranks_ = o.nranks_;
        peer_stats_ = std::move(o.peer_stats_);
        return *this;
    }

    bool enabled() const noexcept {
        return nranks_ > 0;
    }

    PeerStats get_peer_stats(Rank peer) const {
        if (!enabled()) {
            return PeerStats{};
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_stats_.at(peer);
    }

    std::size_t add_peer_comm(Rank peer, std::size_t nbytes) {
        if (!enabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto& p = peer_stats_.at(peer);
        ++p.send_count;
        return p.send_nbytes += nbytes;
    }

    std::string report(int column_width = 12, int label_width = 14) const {
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
        ss << "\n" << std::setw(label_width) << std::left << " - send-total:";
        for (Rank i = 0; i < nranks_; ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(peer_stats_.at(i).send_nbytes) << " ";
        }
        ss << "\n" << std::setw(label_width) << std::left << " - send-mean:";
        for (Rank i = 0; i < nranks_; ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(
                      peer_stats_.at(i).send_nbytes / (double)peer_stats_.at(i).send_count
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
