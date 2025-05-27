/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iomanip>
#include <sstream>

#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

void Statistics::FormatterDefault(std::ostream& os, std::size_t count, double val) {
    os << val;
    if (count > 1) {
        os << " (avg " << (val / count) << ")";
    }
};

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

std::size_t Statistics::add_bytes_stat(std::string const& name, std::size_t nbytes) {
    return add_stat(name, nbytes, [](std::ostream& os, std::size_t count, double val) {
        os << format_nbytes(val);
        if (count > 1) {
            os << " (avg " << format_nbytes(val / count) << ")";
        }
    });
}

Duration Statistics::add_duration_stat(std::string const& name, Duration seconds) {
    return Duration(add_stat(
        name,
        seconds.count(),
        [](std::ostream& os, std::size_t count, double val) {
            os << format_duration(val);
            if (count > 1) {
                os << " (avg " << format_duration(val / count) << ")";
            }
        }
    ));
}

Statistics::MemoryRecorder::MemoryRecorder(
    Statistics* stats, rmm_statistics_resource* mr, std::string name
)
    : stats_{stats}, mr_{mr}, name_{std::move(name)} {
    RAPIDSMPF_EXPECTS(stats_ != nullptr, "the statistics cannot be null");
    RAPIDSMPF_EXPECTS(mr != nullptr, "the memory resource cannot be null");
    mr_->push_counters();
}

Statistics::MemoryRecorder::~MemoryRecorder() {
    if (stats_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(stats_->mutex_);
    auto bytes_counter = mr_->pop_counters().first;
    auto& record = stats_->memory_records_[name_];
    ++record.num_calls;
    record.total += bytes_counter.total;
    record.peak = std::max(record.peak, bytes_counter.peak);
}

std::string Statistics::report(std::string const& header) const {
    std::stringstream ss;
    ss << header;
    if (!enabled()) {
        ss << " disabled.";
        return ss.str();
    }
    std::lock_guard<std::mutex> lock(mutex_);

    std::size_t max_length{0};
    for (auto const& [name, _] : stats_) {
        max_length = std::max(max_length, name.size());
    }
    ss << "\n";
    for (auto const& [name, stat] : stats_) {
        ss << " - " << std::setw(max_length + 3) << std::left << name + ": ";
        stat.formatter()(ss, stat.count(), stat.value());
        ss << "\n";
    }
    ss << "\n";

    // Print memory profiling.
    ss << "Memory Profiling\n";
    ss << "----------------\n";
    if (mr_ == nullptr) {
        ss << "Disabled";
        return ss.str();
    }
    ss << "Legends:\n"
       << "  ncalls - number of times the code block was called.\n"
       << "  peak   - peak memory allocated while running code block.\n"
       << "  total  - total memory allocated while running code block.\n";
    ss << "\nOrdered by: peak (descending)\n\n";

    ss << std::right << std::setw(8) << "ncalls" << std::setw(12) << "peak"
       << std::setw(12) << "total"
       << "  filename:lineno(name)\n";

    // Insert the memory records in a vector we can sort.
    std::vector<std::pair<std::string, rapidsmpf::Statistics::MemoryRecord>>
        sorted_records{memory_records_.begin(), memory_records_.end()};

    // Insert the "main" record, which is the overall statistics from `mr_`.
    sorted_records.emplace_back(
        "main",
        rapidsmpf::Statistics::MemoryRecord{
            .num_calls = 1,
            .total = mr_->get_bytes_counter().total,
            .peak = mr_->get_bytes_counter().peak
        }
    );

    // Sort base on peak memory.
    std::sort(
        sorted_records.begin(),
        sorted_records.end(),
        [](auto const& a, auto const& b) { return a.second.peak > b.second.peak; }
    );

    // Print the sorted records.
    for (auto const& [name, record] : sorted_records) {
        ss << std::right << std::setw(8) << record.num_calls << std::setw(12)
           << rapidsmpf::format_nbytes(record.peak) << std::setw(12)
           << rapidsmpf::format_nbytes(record.total) << "  " << name << "\n";
    }
    return ss.str();
}


}  // namespace rapidsmpf
