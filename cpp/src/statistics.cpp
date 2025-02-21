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

#include <iomanip>
#include <sstream>

#include <rapidsmp/statistics.hpp>

namespace rapidsmp {

Statistics::PeerStats Statistics::get_peer_stats(Rank peer) const {
    if (!enabled()) {
        return PeerStats{};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return peer_stats_.at(peer);
}

std::size_t Statistics::add_payload_send(Rank peer, std::size_t nbytes) {
    if (!enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& p = peer_stats_.at(peer);
    ++p.payload_send_count;
    return p.payload_send_nbytes += nbytes;
}

std::size_t Statistics::add_payload_recv(Rank peer, std::size_t nbytes) {
    if (!enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& p = peer_stats_.at(peer);
    ++p.payload_recv_count;
    return p.payload_recv_nbytes += nbytes;
}

std::string Statistics::report(int column_width, int label_width) const {
    if (!enabled()) {
        return "Statistics: disabled";
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "Statistics:\n";
    ss << std::setw(label_width - 3) << std::left << " - peers:";
    for (Rank i = 0; i < comm_->nranks(); ++i) {
        ss << std::right << std::setw(column_width) << "Rank" << i;
    }
    ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-total:";
    for (Rank i = 0; i < comm_->nranks(); ++i) {
        ss << std::right << std::setw(column_width)
           << format_nbytes(peer_stats_.at(i).payload_send_nbytes) << " ";
    }
    ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-mean:";
    for (Rank i = 0; i < comm_->nranks(); ++i) {
        ss << std::right << std::setw(column_width)
           << format_nbytes(
                  peer_stats_.at(i).payload_send_nbytes
                  / (double)peer_stats_.at(i).payload_send_count
              )
           << " ";
    }
    ss << "\n";
    return ss.str();
}
}  // namespace rapidsmp
