/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iomanip>
#include <sstream>

#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

std::shared_ptr<Statistics> Statistics::disabled() {
    static std::shared_ptr<Statistics> ret = std::make_shared<Statistics>(false);
    return ret;
}

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

bool Statistics::is_memory_profiling_enabled() const {
    return mr_ != nullptr;
}

Statistics::MemoryRecorder::MemoryRecorder(
    Statistics* stats, RmmResourceAdaptor* mr, std::string name
)
    : stats_{stats}, mr_{mr}, name_{std::move(name)} {
    RAPIDSMPF_EXPECTS(stats_ != nullptr, "the statistics cannot be null");
    RAPIDSMPF_EXPECTS(mr != nullptr, "the memory resource cannot be null");
    mr_->begin_scoped_memory_record();
    main_record_ = mr_->get_main_record();
}

Statistics::MemoryRecorder::~MemoryRecorder() {
    if (stats_ == nullptr) {
        return;
    }
    auto const scope = mr_->end_scoped_memory_record();

    std::lock_guard<std::mutex> lock(stats_->mutex_);
    auto& record = stats_->memory_records_[name_];
    ++record.num_calls;
    record.scoped.add_scope(scope);
    record.global_peak = std::max(record.global_peak, scope.peak());
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
    std::vector<std::pair<std::string, MemoryRecord>> sorted_records{
        memory_records_.begin(), memory_records_.end()
    };

    // Sort base on peak memory.
    std::sort(
        sorted_records.begin(),
        sorted_records.end(),
        [](auto const& a, auto const& b) {
            return a.second.scoped.peak() > b.second.scoped.peak();
        }
    );

    // Print the sorted records.
    for (auto const& [name, record] : sorted_records) {
        ss << std::right << std::setw(8) << record.num_calls << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.peak()) << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.total()) << "  " << name << "\n";
    }
    return ss.str();
}

}  // namespace rapidsmpf
