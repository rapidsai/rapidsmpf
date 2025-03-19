/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <sstream>

#include <rapidsmp/shuffler/chunk.hpp>
#include <rapidsmp/shuffler/postbox.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp::shuffler::detail {


void PostBox::insert(Chunk&& chunk) {
    std::lock_guard const lock(mutex_);
    auto [_, inserted] = pigeonhole_[chunk.pid].insert({chunk.cid, std::move(chunk)});
    RAPIDSMP_EXPECTS(inserted, "PostBox.insert(): chunk already exist");
}

Chunk PostBox::extract(PartID pid, ChunkID cid) {
    std::lock_guard const lock(mutex_);
    return extract_item(pigeonhole_.at(pid), cid).second;
}

std::unordered_map<ChunkID, Chunk> PostBox::extract(PartID pid) {
    std::lock_guard const lock(mutex_);
    return extract_value(pigeonhole_, pid);
}

std::vector<Chunk> PostBox::extract_all() {
    std::lock_guard const lock(mutex_);
    std::vector<Chunk> ret;
    for (auto& [_, chunks] : pigeonhole_) {
        for (auto& [_, chunk] : chunks) {
            ret.push_back(std::move(chunk));
        }
    }
    pigeonhole_.clear();
    return ret;
}

bool PostBox::empty() const {
    return pigeonhole_.empty();
}

std::vector<std::tuple<PartID, ChunkID, std::size_t>> PostBox::search(MemoryType mem_type
) const {
    std::lock_guard const lock(mutex_);
    std::vector<std::tuple<PartID, ChunkID, std::size_t>> ret;
    for (auto& [pid, chunks] : pigeonhole_) {
        for (auto& [cid, chunk] : chunks) {
            if (chunk.gpu_data && chunk.gpu_data->mem_type == mem_type) {
                ret.emplace_back(pid, cid, chunk.gpu_data->size);
            }
        }
    }
    return ret;
}

std::string PostBox::str() const {
    if (empty()) {
        return "PostBox()";
    }
    std::stringstream ss;
    ss << "PostBox(";
    for (auto const& [pid, chunks] : pigeonhole_) {
        ss << "p" << pid << ": [";
        for (auto const& [cid, chunk] : chunks) {
            assert(cid == chunk.cid);
            if (chunk.expected_num_chunks) {
                ss << "EOP" << chunk.expected_num_chunks << ", ";
            } else {
                ss << cid << ", ";
            }
        }
        ss << "\b\b], ";
    }
    ss << "\b\b)";
    return ss.str();
}


}  // namespace rapidsmp::shuffler::detail
