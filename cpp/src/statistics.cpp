/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <unordered_set>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace {

bool has_json_unsafe_chars(std::string_view s) {
    return std::ranges::any_of(s, [](unsigned char c) {
        return c == '"' || c == '\\' || c < 0x20;
    });
}

// Pre-computed stat names for record_copy / record_alloc.
struct Names {
    std::string base;  // "alloc-{memtype}" or "copy-{src}-to-{dst}"
    std::string nbytes;  // "{base}-bytes"
    std::string time;  // "{base}-time"
    std::string stream_delay;  // "{base}-stream-delay"
};

using NamesArray = std::array<Names, MEMORY_TYPES.size()>;
using Names2DArray = std::array<NamesArray, MEMORY_TYPES.size()>;

// Predefined render functions.
using FormatterFn = void (*)(std::ostream&, std::vector<Statistics::Stat> const&);

// Formatters indexed by `Statistics::Formatter`. Per-entry rendering description lives on
// the enum `Statistics::Formatter` in statistics.hpp.
constexpr std::array<FormatterFn, static_cast<std::size_t>(Statistics::Formatter::_Count)>
    FORMATTERS = {{
        // Implement `Statistics::Formatter::Default`
        [](std::ostream& os, std::vector<Statistics::Stat> const& s) {
            os << s.at(0).value();
            if (s.at(0).count() > 1) {
                os << " (count " << s.at(0).count() << ")";
            }
        },
        // Implement `Statistics::Formatter::Bytes`
        [](std::ostream& os, std::vector<Statistics::Stat> const& s) {
            auto const val = s.at(0).value();
            auto const count = s.at(0).count();
            os << format_nbytes(val);
            if (count > 1) {
                os << " | avg " << format_nbytes(val / static_cast<double>(count));
            }
        },
        // Implement `Statistics::Formatter::Duration`
        [](std::ostream& os, std::vector<Statistics::Stat> const& s) {
            auto const val = s.at(0).value();
            auto const count = s.at(0).count();
            os << format_duration(val);
            if (count > 1) {
                os << " | avg " << format_duration(val / static_cast<double>(count));
            }
        },
        // Implement `Statistics::Formatter::HitRate`
        [](std::ostream& os, std::vector<Statistics::Stat> const& s) {
            os << s.at(0).value() << "/" << s.at(0).count() << " (hits/lookups)";
        },
        // Implement `Statistics::Formatter::MemoryThroughput`
        [](std::ostream& os, std::vector<Statistics::Stat> const& s) {
            RAPIDSMPF_EXPECTS(
                s.at(0).count() == s.at(1).count() && s.at(1).count() == s.at(2).count(),
                "MemoryThroughput formatter expects the record counters to match"
            );
            os << format_nbytes(s.at(0).value()) << " | "
               << format_duration(s.at(1).value()) << " | "
               << format_nbytes(s.at(0).value() / s.at(1).value()) << "/s"
               << " | avg-stream-delay "
               << format_duration(s.at(2).value() / static_cast<double>(s.at(1).count()));
        },
    }};

}  // namespace

// -- Stat --------------------------------------------------------------------

Statistics::Stat::Stat(std::size_t count, double value, double max)
    : count_{count}, value_{value}, max_{max} {}

void Statistics::Stat::add(double value) {
    ++count_;
    value_ += value;
    max_ = std::max(max_, value);
}

std::size_t Statistics::Stat::count() const noexcept {
    return count_;
}

double Statistics::Stat::value() const noexcept {
    return value_;
}

double Statistics::Stat::max() const noexcept {
    return max_;
}

std::uint8_t* Statistics::Stat::serialize(std::uint8_t* out) const {
    auto const write = [&out](auto const& val) {
        std::memcpy(out, &val, sizeof(val));
        out += sizeof(val);
    };
    write(static_cast<std::uint64_t>(count_));
    write(value_);
    write(max_);
    return out;
}

std::pair<Statistics::Stat, std::span<std::uint8_t const>> Statistics::Stat::deserialize(
    std::span<std::uint8_t const> data
) {
    RAPIDSMPF_EXPECTS(
        data.size() >= serialized_size(),
        "truncated Stat serialization data",
        std::invalid_argument
    );

    // Read a POD value from the front of `buf` and return the remainder.
    auto const read = []<typename T>(std::span<std::uint8_t const> buf, T& val) {
        std::memcpy(&val, buf.data(), sizeof(T));
        return buf.subspan(sizeof(T));
    };
    std::uint64_t count{};
    double value{};
    double max{};
    data = read(data, count);
    data = read(data, value);
    data = read(data, max);
    return {Stat(count, value, max), data};
}

Statistics::Stat Statistics::Stat::merge(Stat const& other) const {
    return Stat(count_ + other.count_, value_ + other.value_, std::max(max_, other.max_));
}

Statistics::~Statistics() noexcept {
    StreamOrderedTiming::cancel_inflight_timings(this);
}

// Leaving `mr_` as std::nullopt disables memory profiling.
Statistics::Statistics(bool enabled) : enabled_{enabled} {}

Statistics::Statistics(
    RmmResourceAdaptor mr, std::shared_ptr<PinnedMemoryResource> pinned_mr
)
    : enabled_{true}, mr_{std::move(mr)}, pinned_mr_{std::move(pinned_mr)} {}

std::shared_ptr<Statistics> Statistics::from_options(
    RmmResourceAdaptor mr,
    config::Options options,
    std::shared_ptr<PinnedMemoryResource> pinned_mr
) {
    bool const statistics = options.get<bool>("statistics", [](auto const& s) {
        return parse_string<bool>(s.empty() ? "False" : s);
    });
    return statistics ? std::make_shared<Statistics>(std::move(mr), std::move(pinned_mr))
                      : Statistics::disabled();
}

std::shared_ptr<Statistics> Statistics::disabled() {
    static std::shared_ptr<Statistics> ret = std::make_shared<Statistics>(false);
    return ret;
}

Statistics::Stat Statistics::get_stat(std::string const& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.at(name);
}

void Statistics::add_stat(std::string const& name, double value) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, _] = stats_.try_emplace(name);
    it->second.add(value);
}

void Statistics::add_report_entry(
    std::string const& report_entry_name,
    std::initializer_list<std::string_view> stat_names,
    Formatter formatter
) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (report_entries_.contains(report_entry_name)) {
        return;
    }
    std::vector<std::string> names(stat_names.begin(), stat_names.end());
    report_entries_.try_emplace(
        report_entry_name,
        ReportEntry{.stat_names = std::move(names), .formatter = formatter}
    );
}

void Statistics::add_report_entry(
    std::string const& report_entry_name,
    std::vector<std::string> stat_names,
    Formatter formatter
) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (report_entries_.contains(report_entry_name)) {
        return;
    }
    report_entries_.try_emplace(
        report_entry_name,
        ReportEntry{.stat_names = std::move(stat_names), .formatter = formatter}
    );
}

void Statistics::add_bytes_stat(std::string const& name, std::size_t nbytes) {
    add_stat(name, static_cast<double>(nbytes));
    add_report_entry(name, {name}, Formatter::Bytes);
}

void Statistics::add_duration_stat(std::string const& name, Duration seconds) {
    add_stat(name, seconds.count());
    add_report_entry(name, {name}, Formatter::Duration);
}

std::vector<std::string> Statistics::list_stat_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ks = std::views::keys(stats_);
    return std::vector<std::string>{ks.begin(), ks.end()};
}

void Statistics::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
}

bool Statistics::is_memory_profiling_enabled() const {
    return mr_.has_value();
}

Statistics::MemoryRecorder::MemoryRecorder(
    Statistics* stats, RmmResourceAdaptor mr, std::string name
)
    : stats_{stats}, mr_{std::move(mr)}, name_{std::move(name)} {
    RAPIDSMPF_EXPECTS(stats_ != nullptr, "the statistics cannot be null");
    mr_->begin_scoped_memory_record();
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

Statistics::MemoryRecorder Statistics::create_memory_recorder(std::string name) {
    if (!mr_.has_value()) {
        return MemoryRecorder{};
    }
    return MemoryRecorder{this, *mr_, std::move(name)};
}

std::unordered_map<std::string, Statistics::MemoryRecord> const&
Statistics::get_memory_records() const {
    return memory_records_;
}

std::string Statistics::report(std::string const& header) const {
    std::stringstream ss;
    ss << header;

    if (!enabled()) {
        ss << " disabled.";
        return ss.str();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Reporting strategy:
    //
    // Each report entry claims one or more stat names and renders them into
    // a single labelled line using the formatter selected by its
    // `Formatter`. Any stat that is not claimed by a report entry is
    // rendered with `Formatter::Default`. All entry lines are sorted
    // alphabetically and printed together.

    using EntryLine = std::pair<std::string, std::string>;

    std::vector<EntryLine> lines;
    std::unordered_set<std::string> consumed;

    auto has_all_stats = [&](auto const& names) {
        return std::ranges::all_of(names, [&](auto const& sname) {
            return stats_.contains(sname);
        });
    };

    for (auto const& [report_entry_name, entry] : report_entries_) {
        if (!has_all_stats(entry.stat_names)) {
            lines.emplace_back(report_entry_name, "No data collected");
            continue;
        }
        std::vector<Stat> args;
        args.reserve(entry.stat_names.size());
        for (auto const& sname : entry.stat_names) {
            args.push_back(stats_.at(sname));
            consumed.insert(sname);
        }
        std::ostringstream line;
        FORMATTERS.at(static_cast<std::size_t>(entry.formatter))(line, args);
        lines.emplace_back(report_entry_name, std::move(line).str());
    }

    // Stats not covered by any report entry fall back to Formatter::Default.
    for (auto const& [name, stat] : stats_) {
        if (consumed.contains(name)) {
            continue;
        }
        std::ostringstream line;
        FORMATTERS.at(static_cast<std::size_t>(Formatter::Default))(line, {stat});
        lines.emplace_back(name, std::move(line).str());
    }

    std::ranges::sort(lines, {}, &EntryLine::first);
    std::size_t max_length = 0;
    for (auto const& [name, _] : lines) {
        max_length = std::max(max_length, name.size());
    }

    ss << "\n";
    for (auto const& [name, text] : lines) {
        ss << " - " << std::setw(safe_cast<int>(max_length) + 3) << std::left
           << name + ": " << text << "\n";
    }
    ss << "\n";

    // Print memory profiling.
    ss << "Memory Profiling\n";
    ss << "----------------\n";
    if (!mr_.has_value()) {
        ss << "Disabled";
        return ss.str();
    }

    // Insert the memory records in a vector we can sort.
    std::vector<std::pair<std::string, MemoryRecord>> sorted_records{
        memory_records_.begin(), memory_records_.end()
    };

    // Insert the "main" record, which is the overall statistics from `mr_`.
    auto const main_record = mr_->get_main_record();
    sorted_records.emplace_back(
        "main (all allocations using RmmResourceAdaptor)",
        MemoryRecord{
            .scoped = main_record, .global_peak = main_record.peak(), .num_calls = 1
        }
    );

    if (pinned_mr_ != PinnedMemoryResource::Disabled) {
        auto const pinned_record = pinned_mr_->get_main_memory_record();
        sorted_records.emplace_back(
            "main (all allocations using PinnedMemoryResource)",
            MemoryRecord{
                .scoped = pinned_record,
                .global_peak = pinned_record.peak(),
                .num_calls = 1
            }
        );
    }

    // Sort based on peak memory.
    std::ranges::sort(sorted_records, [](auto const& a, auto const& b) {
        return a.second.scoped.peak() > b.second.scoped.peak();
    });
    ss << "Legends:\n"
       << "  ncalls - number of times the scope was executed.\n"
       << "  peak   - peak memory usage by the scope.\n"
       << "  g-peak - global peak memory usage during the scope's execution.\n"
       << "  accum  - total accumulated memory allocations by the scope.\n"
       << "  max    - largest single allocation by the scope.\n";
    ss << "\nOrdered by: peak (descending)\n\n";

    ss << std::right << std::setw(8) << "ncalls" << std::setw(12) << "peak"
       << std::setw(12) << "g-peak" << std::setw(12) << "accum" << std::setw(12) << "max"
       << "  filename:lineno(name)\n";

    // Print the sorted records.
    for (auto const& [name, record] : sorted_records) {
        ss << std::right << std::setw(8) << record.num_calls << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.peak()) << std::setw(12)
           << rapidsmpf::format_nbytes(record.global_peak) << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.total()) << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.max()) << "  " << name << "\n";
    }
    ss << "\nLimitation:\n"
       << "  - A scope only tracks allocations made by the thread that entered it.\n";
    return ss.str();
}

void Statistics::write_json(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto const check_name = [](std::string_view name, char const* context) {
        RAPIDSMPF_EXPECTS(
            !has_json_unsafe_chars(name),
            std::string(context)
                + " cannot contain characters that require JSON escaping: "
                + std::string(name),
            std::invalid_argument
        );
    };
    for (auto const& [name, _] : stats_) {
        check_name(name, "stat name");
    }
    for (auto const& [name, _] : memory_records_) {
        check_name(name, "memory record name");
    }

    os << "{\n";
    os << "  \"statistics\": {";
    for (std::string sep; auto const& [name, stat] : stats_) {
        os << std::exchange(sep, ",") << "\n    \"" << name << "\": {"
           << "\"count\": " << stat.count() << ", "
           << "\"value\": " << stat.value() << ", "
           << "\"max\": " << stat.max() << "}";
    }
    os << (stats_.empty() ? "" : "\n  ") << "}";

    if (!memory_records_.empty()) {
        // Sort by name for deterministic output (unordered_map has no order).
        std::vector<std::string> names;
        names.reserve(memory_records_.size());
        for (auto const& [n, _] : memory_records_)
            names.push_back(n);
        std::ranges::sort(names);

        os << ",\n  \"memory_records\": {";
        for (std::string sep; auto const& n : names) {
            auto const& rec = memory_records_.at(n);
            os << std::exchange(sep, ",") << "\n    \"" << n << "\": {"
               << "\"num_calls\": " << rec.num_calls << ", "
               << "\"peak_bytes\": " << rec.scoped.peak() << ", "
               << "\"total_bytes\": " << rec.scoped.total() << ", "
               << "\"global_peak_bytes\": " << rec.global_peak << "}";
        }
        os << "\n  }";
    }
    os << "\n}\n";
}

void Statistics::write_json(std::filesystem::path const& filepath) const {
    std::ofstream f(filepath);
    RAPIDSMPF_EXPECTS(
        f.is_open(), "Cannot open file: " + filepath.string(), std::ios_base::failure
    );
    write_json(f);
    RAPIDSMPF_EXPECTS(
        !f.fail(), "Failed writing to: " + filepath.string(), std::ios_base::failure
    );
}

std::shared_ptr<Statistics> Statistics::copy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ret = std::make_shared<Statistics>(enabled_.load(std::memory_order_acquire));
    ret->stats_ = stats_;
    ret->report_entries_ = report_entries_;
    return ret;
}

namespace {
// POD writer/reader helpers shared by serialize/deserialize.
void write_pod(std::uint8_t*& ptr, auto const& val) {
    std::memcpy(ptr, &val, sizeof(val));
    ptr += sizeof(val);
}

template <typename T>
std::span<std::uint8_t const> read_pod(std::span<std::uint8_t const> buf, T& val) {
    RAPIDSMPF_EXPECTS(
        buf.size() >= sizeof(T),
        "truncated Statistics serialization data",
        std::invalid_argument
    );
    std::memcpy(&val, buf.data(), sizeof(T));
    return buf.subspan(sizeof(T));
}
}  // namespace

std::vector<std::uint8_t> Statistics::serialize() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Binary layout:
    //   [enabled: uint8]
    //   [num_stats: uint64]
    //   Per stat (sorted by name): [name_len: uint64] [name] [Stat payload]
    //   [num_entries: uint64]
    //   Per entry (sorted by name): [name_len: uint64] [name]
    //                               [formatter: uint8]
    //                               [num_stat_names: uint64]
    //                               Per stat_name: [len: uint64] [bytes]
    std::size_t total = sizeof(std::uint8_t);  // enabled flag
    total += sizeof(std::uint64_t);
    for (auto const& [name, stat] : stats_) {
        total += sizeof(std::uint64_t) + name.size() + Stat::serialized_size();
    }
    total += sizeof(std::uint64_t);
    for (auto const& [name, entry] : report_entries_) {
        total += sizeof(std::uint64_t) + name.size();
        total += sizeof(std::uint8_t);
        total += sizeof(std::uint64_t);
        for (auto const& sn : entry.stat_names) {
            total += sizeof(std::uint64_t) + sn.size();
        }
    }

    std::vector<std::uint8_t> buf(total);
    std::uint8_t* ptr = buf.data();

    write_pod(ptr, static_cast<std::uint8_t>(enabled_.load(std::memory_order_acquire)));
    write_pod(ptr, static_cast<std::uint64_t>(stats_.size()));
    for (auto const& [name, stat] : stats_) {
        write_pod(ptr, static_cast<std::uint64_t>(name.size()));
        std::memcpy(ptr, name.data(), name.size());
        ptr += name.size();
        ptr = stat.serialize(ptr);
    }

    write_pod(ptr, static_cast<std::uint64_t>(report_entries_.size()));
    for (auto const& [name, entry] : report_entries_) {
        write_pod(ptr, static_cast<std::uint64_t>(name.size()));
        std::memcpy(ptr, name.data(), name.size());
        ptr += name.size();
        write_pod(ptr, static_cast<std::uint8_t>(entry.formatter));
        write_pod(ptr, static_cast<std::uint64_t>(entry.stat_names.size()));
        for (auto const& sn : entry.stat_names) {
            write_pod(ptr, static_cast<std::uint64_t>(sn.size()));
            std::memcpy(ptr, sn.data(), sn.size());
            ptr += sn.size();
        }
    }
    return buf;
}

std::shared_ptr<Statistics> Statistics::deserialize(std::span<std::uint8_t const> data) {
    auto const read_string = [&](std::uint64_t len) {
        RAPIDSMPF_EXPECTS(
            data.size() >= len,
            "truncated Statistics serialization data",
            std::invalid_argument
        );
        std::string s(reinterpret_cast<char const*>(data.data()), len);
        data = data.subspan(len);
        return s;
    };

    std::uint8_t enabled{};
    data = read_pod(data, enabled);
    auto ret = std::make_shared<Statistics>(enabled != 0);

    std::uint64_t num_stats{};
    data = read_pod(data, num_stats);
    for (std::uint64_t i = 0; i < num_stats; ++i) {
        std::uint64_t name_len{};
        data = read_pod(data, name_len);
        auto name = read_string(name_len);
        auto [stat, remaining] = Stat::deserialize(data);
        data = remaining;
        ret->stats_.emplace(std::move(name), std::move(stat));
    }

    std::uint64_t num_entries{};
    data = read_pod(data, num_entries);
    for (std::uint64_t i = 0; i < num_entries; ++i) {
        std::uint64_t name_len{};
        data = read_pod(data, name_len);
        auto name = read_string(name_len);

        std::uint8_t fmt{};
        data = read_pod(data, fmt);
        RAPIDSMPF_EXPECTS(
            fmt < static_cast<std::uint8_t>(Formatter::_Count),
            "Statistics::deserialize: Formatter value out of range",
            std::invalid_argument
        );
        auto const formatter = static_cast<Formatter>(fmt);

        std::uint64_t num_stat_names{};
        data = read_pod(data, num_stat_names);
        std::vector<std::string> stat_names;
        stat_names.reserve(num_stat_names);
        for (std::uint64_t j = 0; j < num_stat_names; ++j) {
            std::uint64_t sn_len{};
            data = read_pod(data, sn_len);
            stat_names.push_back(read_string(sn_len));
        }
        ret->report_entries_.emplace(
            std::move(name),
            ReportEntry{.stat_names = std::move(stat_names), .formatter = formatter}
        );
    }
    return ret;
}

std::shared_ptr<Statistics> Statistics::merge(
    std::span<std::shared_ptr<Statistics> const> stats
) {
    RAPIDSMPF_EXPECTS(
        !stats.empty(),
        "Statistics::merge: input span must not be empty",
        std::invalid_argument
    );

    // Snapshot each input under its own mutex. Folding the snapshots
    // afterwards avoids having to hold multiple mutexes at once.
    struct Snapshot {
        std::map<std::string, Stat> stats;
        std::map<std::string, ReportEntry> entries;
        bool enabled;
    };

    std::vector<Snapshot> snapshots;
    snapshots.reserve(stats.size());
    for (auto const& s : stats) {
        RAPIDSMPF_EXPECTS(
            s != nullptr,
            "Statistics::merge: pointer must not be null",
            std::invalid_argument
        );
        std::lock_guard lock(s->mutex_);
        snapshots.push_back({s->stats_, s->report_entries_, s->enabled()});
    }

    bool const any_enabled =
        std::ranges::any_of(snapshots, [](auto const& s) { return s.enabled; });
    auto ret = std::make_shared<Statistics>(any_enabled);

    for (auto const& snap : snapshots) {
        for (auto const& [name, stat] : snap.stats) {
            auto [it, inserted] = ret->stats_.try_emplace(name, stat);
            if (!inserted) {
                it->second = it->second.merge(stat);
            }
        }
        for (auto const& [name, entry] : snap.entries) {
            auto [it, inserted] = ret->report_entries_.try_emplace(name, entry);
            if (!inserted) {
                RAPIDSMPF_EXPECTS(
                    it->second.formatter == entry.formatter
                        && it->second.stat_names == entry.stat_names,
                    "Statistics::merge: report entry '" + name
                        + "' has conflicting formatter or stat_names",
                    std::invalid_argument
                );
            }
        }
    }
    return ret;
}

void Statistics::record_copy(
    MemoryType src, MemoryType dst, std::size_t nbytes, StreamOrderedTiming&& timing
) {
    // Construct all stat names once, at first call.
    static Names2DArray const name_map = [] {
        Names2DArray ret;
        for (MemoryType s : MEMORY_TYPES) {
            auto const src_name = to_lower(to_string(s));
            for (MemoryType d : MEMORY_TYPES) {
                auto const dst_name = to_lower(to_string(d));
                auto base = "copy-" + src_name + "-to-" + dst_name;
                ret[static_cast<std::size_t>(s)][static_cast<std::size_t>(d)] = Names{
                    .base = base,
                    .nbytes = base + "-bytes",
                    .time = base + "-time",
                    .stream_delay = base + "-stream-delay",
                };
            }
        }
        return ret;
    }();
    auto const& names =
        name_map[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)];

    timing.stop_and_record(names.time, names.stream_delay);
    add_stat(names.nbytes, static_cast<double>(nbytes));
    add_report_entry(
        names.base,
        {names.nbytes, names.time, names.stream_delay},
        Formatter::MemoryThroughput
    );
}

void Statistics::record_alloc(
    MemoryType mem_type, std::size_t nbytes, StreamOrderedTiming&& timing
) {
    // Construct all stat names once, at first call.
    static NamesArray const names = [] {
        NamesArray ret;
        for (MemoryType mt : MEMORY_TYPES) {
            auto base = "alloc-" + to_lower(to_string(mt));
            ret[static_cast<std::size_t>(mt)] = Names{
                .base = base,
                .nbytes = base + "-bytes",
                .time = base + "-time",
                .stream_delay = base + "-stream-delay",
            };
        }
        return ret;
    }();

    auto const& n = names[static_cast<std::size_t>(mem_type)];

    timing.stop_and_record(n.time, n.stream_delay);
    add_stat(n.nbytes, static_cast<double>(nbytes));
    add_report_entry(
        n.base, {n.nbytes, n.time, n.stream_delay}, Formatter::MemoryThroughput
    );
}

}  // namespace rapidsmpf
