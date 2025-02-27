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

#include <sstream>

#include <rapidsmp/statistics.hpp>

namespace rapidsmp {


Statistics::Stat Statistics::get_stat(std::string const& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.at(name);
}

double Statistics::add_stat(
    std::string const& name, double value, Formatter const& formatter
) {
    if (!enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(name);
    if (it == stats_.end()) {
        it = stats_.insert({name, Stat(formatter)}).first;
    }
    return it->second.add(value);
}

std::size_t Statistics::add_bytes_stat(
    std::string const& name, std::size_t nbytes, bool with_average
) {
    if (with_average) {
        return add_stat(
            name,
            nbytes,
            [](std::ostream& os, std::size_t count, double val) {
                os << format_nbytes(val) << " (avg " << format_nbytes(val / count) << ")";
            }
        );
    } else {
        return add_stat(
            name,
            nbytes,
            [](std::ostream& os, std::size_t count, double val) {
                os << format_nbytes(val);
            }
        );
    }
}

std::string Statistics::report(int column_width, int label_width) const {
    if (!enabled()) {
        return "Statistics: disabled";
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "Statistics:\n";
    for (auto const& [name, stat] : stats_) {
        ss << "  - " << name << ": ";
        stat.formatter_(ss, stat.count_, stat.value_);
        ss << "\n";
    }
    return ss.str();
}
}  // namespace rapidsmp
