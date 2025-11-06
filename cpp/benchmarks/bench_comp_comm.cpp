/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <glob.h>
#include <mpi.h>
#include <unistd.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

#include "utils/misc.hpp"
#include "utils/rmm_stack.hpp"

// nvCOMP managers (v3.x API)
#include <nvcomp/cascaded.hpp>
#include <nvcomp/lz4.hpp>

using namespace rapidsmpf;

namespace {

enum class PackMode {
    Table,
    Columns
};
enum class Algo {
    Cascaded,
    LZ4
};

struct KvParams {
    // Common
    std::size_t chunk_size{1 << 20};
    // Cascaded
    int cascaded_rle{1};
    int cascaded_delta{1};
    int cascaded_bitpack{1};
};

struct Args {
    std::string comm_type{"mpi"};
    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::string rmm_mr{"pool"};
    std::string file_pattern;  // required
    PackMode pack_mode{PackMode::Table};
    Algo algo{Algo::Cascaded};
    KvParams params{};
    std::uint64_t num_ops{1};
    bool enable_cupti_monitoring{false};
    std::string cupti_csv_prefix;
};

std::vector<std::string> expand_glob(std::string const& pattern) {
    std::vector<std::string> files;
    glob_t glob_result{};
    int rc = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    if (rc == 0) {
        for (std::size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.emplace_back(glob_result.gl_pathv[i]);
        }
    }
    globfree(&glob_result);
    std::sort(files.begin(), files.end());
    return files;
}

std::size_t parse_nbytes(std::string const& s) {
    // Simple parser: supports suffixes KiB, MiB, GiB, KB, MB, GB, or no suffix.
    auto to_lower = [](char c) { return static_cast<char>(std::tolower(c)); };
    std::string v;
    v.reserve(s.size());
    for (char c : s)
        v.push_back(to_lower(c));

    std::size_t mult = 1;
    if (v.ends_with("kib")) {
        mult = 1ull << 10;
        v = v.substr(0, v.size() - 3);
    } else if (v.ends_with("mib")) {
        mult = 1ull << 20;
        v = v.substr(0, v.size() - 3);
    } else if (v.ends_with("gib")) {
        mult = 1ull << 30;
        v = v.substr(0, v.size() - 3);
    } else if (v.ends_with("kb")) {
        mult = 1000ull;
        v = v.substr(0, v.size() - 2);
    } else if (v.ends_with("mb")) {
        mult = 1000ull * 1000ull;
        v = v.substr(0, v.size() - 2);
    } else if (v.ends_with("gb")) {
        mult = 1000ull * 1000ull * 1000ull;
        v = v.substr(0, v.size() - 2);
    }

    return static_cast<std::size_t>(std::stoll(v)) * mult;
}

KvParams parse_kv_params(std::string const& kv) {
    KvParams p{};
    if (kv.empty())
        return p;
    std::size_t start = 0;
    while (start < kv.size()) {
        auto comma = kv.find(',', start);
        auto part = kv.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start
        );
        auto eq = part.find('=');
        if (eq != std::string::npos) {
            std::string key = part.substr(0, eq);
            std::string val = part.substr(eq + 1);
            if (key == "chunk_size")
                p.chunk_size = parse_nbytes(val);
            else if (key == "delta")
                p.cascaded_delta = std::stoi(val);
            else if (key == "rle")
                p.cascaded_rle = std::stoi(val);
            else if (key == "bitpack")
                p.cascaded_bitpack = std::stoi(val);
        }
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return p;
}

struct PhaseThroughputs {
    double compress_Bps{0.0};
    double decompress_Bps{0.0};
    double comp_send_Bps{0.0};
    double recv_decomp_Bps{0.0};
    double send_only_Bps{0.0};
    double recv_only_Bps{0.0};
};

class NvcompCodec {
  public:
    virtual ~NvcompCodec() = default;
    virtual std::size_t get_max_compressed_bytes(std::size_t uncompressed_bytes) = 0;
    virtual void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream
    ) = 0;
    virtual void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) = 0;
};

class LZ4Codec final : public NvcompCodec {
  public:
    explicit LZ4Codec(std::size_t chunk_size) : chunk_size_{chunk_size} {}

    std::size_t get_max_compressed_bytes(std::size_t in_bytes) override {
        nvcomp::LZ4Manager mgr{static_cast<int>(chunk_size_), 0};
        auto cfg = mgr.configure_compression(in_bytes);
        return cfg.max_compressed_buffer_size;
    }

    void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        nvcomp::LZ4Manager mgr{static_cast<int>(chunk_size_), 0, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        mgr.compress(d_out, d_in, cfg);
        // Compressed size is stored at the beginning of output; ask manager
        auto info = mgr.get_compress_result(d_out);
        *out_bytes = info.compressed_bytes;
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        nvcomp::LZ4Manager mgr{static_cast<int>(chunk_size_), 0, stream.value()};
        auto cfg = mgr.configure_decompression(d_in, in_bytes);
        (void)out_bytes;  // decomp size implied by cfg
        mgr.decompress(d_out, d_in, cfg);
    }

  private:
    std::size_t chunk_size_;
};

class CascadedCodec final : public NvcompCodec {
  public:
    CascadedCodec(std::size_t chunk_size, int rle, int delta, int bitpack)
        : opts_{rle != 0, delta != 0, bitpack != 0, static_cast<int>(chunk_size)} {}

    std::size_t get_max_compressed_bytes(std::size_t in_bytes) override {
        nvcomp::CascadedManager mgr{opts_};
        auto cfg = mgr.configure_compression(in_bytes);
        return cfg.max_compressed_buffer_size;
    }

    void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        nvcomp::CascadedManager mgr{opts_, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        mgr.compress(d_out, d_in, cfg);
        auto info = mgr.get_compress_result(d_out);
        *out_bytes = info.compressed_bytes;
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        nvcomp::CascadedManager mgr{opts_, stream.value()};
        auto cfg = mgr.configure_decompression(d_in, in_bytes);
        (void)out_bytes;
        mgr.decompress(d_out, d_in, cfg);
    }

  private:
    nvcomp::CascadedOptions opts_{};
};

std::unique_ptr<NvcompCodec> make_codec(Algo algo, KvParams const& p) {
    switch (algo) {
    case Algo::LZ4:
        return std::make_unique<LZ4Codec>(p.chunk_size);
    case Algo::Cascaded:
    default:
        return std::make_unique<CascadedCodec>(
            p.chunk_size, p.cascaded_rle, p.cascaded_delta, p.cascaded_bitpack
        );
    }
}

// Convenience: wrap metadata + gpu_data into rapidsmpf::PackedData
static std::unique_ptr<PackedData> pack_table_to_packed(
    cudf::table_view tv, rmm::cuda_stream_view stream, BufferResource* br
) {
    auto packed = cudf::pack(tv, stream, br->device_mr());
    auto metadata =
        std::make_unique<std::vector<std::uint8_t>>(std::move(packed.metadata));
    auto buf = br->move(
        std::make_unique<rmm::device_buffer>(std::move(packed.gpu_data)), stream
    );
    return std::make_unique<PackedData>(std::move(metadata), std::move(buf));
}

struct ArgumentParser {
    ArgumentParser(int argc, char* const* argv, bool use_mpi) {
        int rank = 0;
        if (use_mpi) {
            RAPIDSMPF_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");
            RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        }
        try {
            int opt;
            // C: comm, r: runs, w: warmups, m: rmm, F: files, P: pack mode, A: algo, K:
            // kv, p: ops, M: cupti, h: help
            while ((opt = getopt(argc, argv, "C:r:w:m:F:P:A:K:p:M:h")) != -1) {
                switch (opt) {
                case 'C':
                    args_.comm_type = std::string{optarg};
                    break;
                case 'r':
                    parse_integer(args_.num_runs, optarg);
                    break;
                case 'w':
                    parse_integer(args_.num_warmups, optarg);
                    break;
                case 'm':
                    args_.rmm_mr = std::string{optarg};
                    break;
                case 'F':
                    args_.file_pattern = std::string{optarg};
                    break;
                case 'P':
                    {
                        std::string v{optarg};
                        if (v == "table")
                            args_.pack_mode = PackMode::Table;
                        else if (v == "columns")
                            args_.pack_mode = PackMode::Columns;
                        else
                            RAPIDSMPF_FAIL(
                                "-P must be one of {table, columns}",
                                std::invalid_argument
                            );
                        break;
                    }
                case 'A':
                    {
                        std::string v{optarg};
                        if (v == "cascaded")
                            args_.algo = Algo::Cascaded;
                        else if (v == "lz4")
                            args_.algo = Algo::LZ4;
                        else
                            RAPIDSMPF_FAIL(
                                "-A must be one of {cascaded, lz4}", std::invalid_argument
                            );
                        break;
                    }
                case 'K':
                    args_.params = parse_kv_params(std::string{optarg});
                    break;
                case 'p':
                    parse_integer(args_.num_ops, optarg);
                    break;
                case 'M':
                    args_.enable_cupti_monitoring = true;
                    args_.cupti_csv_prefix = std::string{optarg};
                    break;
                case 'h':
                default:
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>    {mpi, ucxx} (default: mpi)\n"
                           << "  -r <num>     Number of runs (default: 1)\n"
                           << "  -w <num>     Number of warmup runs (default: 0)\n"
                           << "  -m <mr>      RMM MR {cuda, pool, async, managed} "
                              "(default: pool)\n"
                           << "  -F <pattern> Parquet file glob/pattern (required)\n"
                           << "  -P <mode>    Packing mode {table, columns} (default: "
                              "table)\n"
                           << "  -A <algo>    {cascaded, lz4} (default: cascaded)\n"
                           << "  -K <kv>      Algo params, e.g. "
                              "chunk_size=1MiB,delta=1,rle=1,bitpack=1\n"
                           << "  -p <num>     Number of concurrent ops (default: 1)\n"
                           << "  -M <path>    CUPTI CSV path prefix (enable CUPTI)\n"
                           << "  -h           Show this help\n";
                        if (rank == 0)
                            std::cerr << ss.str();
                        if (use_mpi)
                            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                        else
                            std::exit(0);
                    }
                }
            }
        } catch (std::exception const& e) {
            std::cerr << "Error parsing arguments: " << e.what() << std::endl;
            if (use_mpi)
                RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            else
                std::exit(-1);
        }
        if (args_.file_pattern.empty()) {
            std::cerr << "-F <pattern> is required" << std::endl;
            if (use_mpi)
                RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            else
                std::exit(-1);
        }
        if (args_.rmm_mr == "cuda") {
            if (rank == 0) {
                std::cout << "WARNING: using the default cuda memory resource (-m cuda) "
                             "might leak memory!"
                          << std::endl;
            }
        }
    }

    Args const& get() const {
        return args_;
    }

  private:
    Args args_{};
};

struct PackedItem {
    // Ownership: we store size and buffer pointer for the packed payload
    std::unique_ptr<PackedData> packed;  // original packed cudf table/column
};

struct BuffersToSend {
    // For each op, we will send these items
    std::vector<PackedItem> items;
    std::size_t total_uncompressed_bytes{0};
};

BuffersToSend make_packed_items(
    cudf::table const& table,
    PackMode mode,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    BuffersToSend ret{};
    if (mode == PackMode::Table) {
        auto item = PackedItem{};
        item.packed = pack_table_to_packed(table.view(), stream, br);
        ret.total_uncompressed_bytes +=
            item.packed->data->size + item.packed->metadata->size();
        ret.items.emplace_back(std::move(item));
    } else {
        auto tv = table.view();
        for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
            cudf::table_view col_tv{std::vector<cudf::column_view>{tv.column(i)}};
            auto item = PackedItem{};
            item.packed = pack_table_to_packed(col_tv, stream, br);
            ret.total_uncompressed_bytes +=
                item.packed->data->size + item.packed->metadata->size();
            ret.items.emplace_back(std::move(item));
        }
    }
    return ret;
}

// Send/recv helpers: send a header (compressed size) as host buffer.
struct SizeHeader {
    std::uint64_t bytes;
};

struct Timings {
    double compress_s{0.0};
    double decompress_s{0.0};
    double comp_send_s{0.0};
    double recv_decomp_s{0.0};
    double send_only_s{0.0};
    double recv_only_s{0.0};
};

// Returns timings and bytes counters
struct Counters {
    std::size_t logical_uncompressed_bytes{0};
    std::size_t logical_compressed_bytes{0};
};

struct RunResult {
    Timings times;
    Counters counts;
};

RunResult run_once(
    std::shared_ptr<Communicator> const& comm,
    Args const& args,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> const& statistics,
    BuffersToSend const& data,
    NvcompCodec& codec
) {
    auto const nranks = comm->nranks();
    auto const rank = comm->rank();
    auto const dst = static_cast<Rank>((rank + 1) % nranks);
    auto const src = static_cast<Rank>((rank - 1 + nranks) % nranks);

    Tag tag_size{1, 0};
    Tag tag_payload{1, 1};
    Tag tag_nocomp{2, 0};

    // Clone packed items into raw device buffers for repeated ops
    std::vector<std::unique_ptr<Buffer>> nocomp_payloads;
    nocomp_payloads.reserve(data.items.size());
    for (auto const& it : data.items) {
        // Copy metadata + data into a contiguous device buffer for pure send path?
        // For pure send/recv, we only send the device payload; metadata isn't needed for
        // metrics. We'll send the packed->data buffer.
        auto reservation = br->reserve_or_fail(it.packed->data->size, MemoryType::DEVICE);
        auto buf = br->allocate(it.packed->data->size, stream, reservation);
        buffer_copy(*buf, *it.packed->data, it.packed->data->size);
        nocomp_payloads.emplace_back(std::move(buf));
    }

    // Pre-allocate compression outputs for each item
    std::vector<std::unique_ptr<Buffer>> comp_outputs;
    std::vector<std::size_t> comp_output_sizes(data.items.size());
    comp_outputs.reserve(data.items.size());
    for (std::size_t i = 0; i < data.items.size(); ++i) {
        auto const in_bytes = data.items[i].packed->data->size;
        auto const max_out = codec.get_max_compressed_bytes(in_bytes);
        auto reservation = br->reserve_or_fail(max_out, MemoryType::DEVICE);
        comp_outputs.emplace_back(br->allocate(max_out, stream, reservation));
    }

    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto t0 = Clock::now();
    // Compress all items (single batch) on stream
    for (std::size_t i = 0; i < data.items.size(); ++i) {
        void const* d_in = data.items[i].packed->data->device()->get()->data();
        void* d_out = comp_outputs[i]->device()->get()->data();
        std::size_t out_bytes = 0;
        codec.compress(d_in, data.items[i].packed->data->size, d_out, &out_bytes, stream);
        comp_output_sizes[i] = out_bytes;
    }
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto t1 = Clock::now();

    // Phase A: pure send/recv (no compression)
    auto a0 = Clock::now();
    std::vector<std::unique_ptr<Communicator::Future>> send_futs;
    std::vector<std::unique_ptr<Communicator::Future>> recv_futs;
    send_futs.reserve(args.num_ops * nocomp_payloads.size());
    recv_futs.reserve(args.num_ops * nocomp_payloads.size());

    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (std::size_t i = 0; i < nocomp_payloads.size(); ++i) {
            // post recv first
            if (src != rank) {
                auto res =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto recv_buf = br->allocate(nocomp_payloads[i]->size, stream, res);
                recv_futs.push_back(comm->recv(src, tag_nocomp, std::move(recv_buf)));
            }
        }
        for (std::size_t i = 0; i < nocomp_payloads.size(); ++i) {
            if (dst != rank) {
                auto res =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto send_buf = br->allocate(nocomp_payloads[i]->size, stream, res);
                buffer_copy(*send_buf, *nocomp_payloads[i], nocomp_payloads[i]->size);
                send_futs.push_back(comm->send(std::move(send_buf), dst, tag_nocomp));
            }
        }
    }
    while (!send_futs.empty()) {
        std::ignore = comm->test_some(send_futs);
    }
    auto a1 = Clock::now();
    while (!recv_futs.empty()) {
        std::ignore = comm->test_some(recv_futs);
    }
    auto a2 = Clock::now();

    // Phase B: compressed path (send size header, then compressed payload)
    auto b0 = Clock::now();
    std::vector<std::unique_ptr<Communicator::Future>> send_hdr_futs;
    std::vector<std::unique_ptr<Communicator::Future>> send_cmp_futs;
    std::vector<std::unique_ptr<Communicator::Future>> recv_hdr_futs;
    std::vector<std::unique_ptr<Communicator::Future>> recv_cmp_futs;
    send_hdr_futs.reserve(args.num_ops * data.items.size());
    send_cmp_futs.reserve(args.num_ops * data.items.size());
    recv_hdr_futs.reserve(args.num_ops * data.items.size());
    recv_cmp_futs.reserve(args.num_ops * data.items.size());

    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (std::size_t i = 0; i < data.items.size(); ++i) {
            // post recv header
            if (src != rank) {
                auto res_h = br->reserve_or_fail(sizeof(SizeHeader), MemoryType::HOST);
                auto hdr = br->allocate(sizeof(SizeHeader), stream, res_h);
                recv_hdr_futs.push_back(comm->recv(src, tag_size, std::move(hdr)));
            }
        }
        for (std::size_t i = 0; i < data.items.size(); ++i) {
            if (dst != rank) {
                auto res_h = br->reserve_or_fail(sizeof(SizeHeader), MemoryType::HOST);
                auto hdr = br->allocate(sizeof(SizeHeader), stream, res_h);
                // write header
                hdr->write_access([&](std::byte* p, rmm::cuda_stream_view) {
                    SizeHeader h{static_cast<std::uint64_t>(comp_output_sizes[i])};
                    std::memcpy(p, &h, sizeof(SizeHeader));
                });
                send_hdr_futs.push_back(comm->send(std::move(hdr), dst, tag_size));
            }
        }
    }
    while (!send_hdr_futs.empty()) {
        std::ignore = comm->test_some(send_hdr_futs);
    }
    while (!recv_hdr_futs.empty()) {
        std::ignore = comm->test_some(recv_hdr_futs);
    }

    // Post payload recvs now that we know sizes
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (std::size_t i = 0; i < data.items.size(); ++i) {
            if (src != rank) {
                // reuse comp_output_sizes[i] as expected size since peers are symmetric
                auto res = br->reserve_or_fail(comp_output_sizes[i], MemoryType::DEVICE);
                auto buf = br->allocate(comp_output_sizes[i], stream, res);
                recv_cmp_futs.push_back(comm->recv(src, tag_payload, std::move(buf)));
            }
        }
        for (std::size_t i = 0; i < data.items.size(); ++i) {
            if (dst != rank) {
                // send compressed
                auto tmp_buf =
                    br->reserve_or_fail(comp_output_sizes[i], MemoryType::DEVICE);
                auto send_buf = br->allocate(comp_output_sizes[i], stream, tmp_buf);
                buffer_copy(*send_buf, *comp_outputs[i], comp_output_sizes[i]);
                send_cmp_futs.push_back(
                    comm->send(std::move(send_buf), dst, tag_payload)
                );
            }
        }
    }
    while (!send_cmp_futs.empty()) {
        std::ignore = comm->test_some(send_cmp_futs);
    }
    auto b1 = Clock::now();
    while (!recv_cmp_futs.empty()) {
        std::ignore = comm->test_some(recv_cmp_futs);
    }
    auto b2 = Clock::now();

    // Decompress received buffers (simulate by decompressing our own produced outputs in
    // symmetric setup)
    auto c0 = Clock::now();
    for (std::size_t i = 0; i < data.items.size(); ++i) {
        auto const out_bytes = data.items[i].packed->data->size;
        auto res = br->reserve_or_fail(out_bytes, MemoryType::DEVICE);
        auto out = br->allocate(out_bytes, stream, res);
        void const* d_in = comp_outputs[i]->device()->get()->data();
        void* d_out = out->device()->get()->data();
        codec.decompress(d_in, comp_output_sizes[i], d_out, out_bytes, stream);
    }
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto c1 = Clock::now();

    RunResult result{};
    result.times.compress_s = std::chrono::duration<double>(t1 - t0).count();
    result.times.send_only_s = std::chrono::duration<double>(a1 - a0).count();
    result.times.recv_only_s = std::chrono::duration<double>(a2 - a1).count();
    result.times.comp_send_s = std::chrono::duration<double>(b1 - b0).count();
    result.times.recv_decomp_s = std::chrono::duration<double>(b2 - b1).count()
                                 + std::chrono::duration<double>(c1 - c0).count();
    result.times.decompress_s = std::chrono::duration<double>(c1 - c0).count();

    result.counts.logical_uncompressed_bytes =
        data.total_uncompressed_bytes * args.num_ops;
    result.counts.logical_compressed_bytes =
        std::accumulate(
            comp_output_sizes.begin(), comp_output_sizes.end(), std::size_t{0}
        )
        * args.num_ops;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    // Check if we should use bootstrap mode with rrun
    bool use_bootstrap = std::getenv("RAPIDSMPF_RANK") != nullptr;

    int provided = 0;
    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
        RAPIDSMPF_EXPECTS(
            provided == MPI_THREAD_MULTIPLE,
            "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
        );
    }

    ArgumentParser parser{argc, argv, !use_bootstrap};
    Args const& args = parser.get();

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        if (use_bootstrap) {
            std::cerr << "Error: MPI communicator requires MPI initialization. Don't use "
                         "with rrun or unset RAPIDSMPF_RANK."
                      << std::endl;
            return 1;
        }
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, options);
    } else if (args.comm_type == "ucxx") {
        if (use_bootstrap) {
            comm = rapidsmpf::bootstrap::create_ucxx_comm(
                rapidsmpf::bootstrap::Backend::AUTO, options
            );
        } else {
            comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options);
        }
    } else {
        std::cerr << "Unknown communicator: " << args.comm_type << std::endl;
        return 1;
    }

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();

    // RMM setup
    auto const mr_stack = set_current_rmm_stack(args.rmm_mr);
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    BufferResource br{mr};

    // Hardware info
    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');
        RAPIDSMPF_CUDA_TRY(
            cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev)
        );
        cudaDeviceProp properties;
        RAPIDSMPF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Hardware setup: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    Device number: " << cur_dev << "\n";
        ss << "    PCI Bus ID: " << pci_bus_id.substr(0, pci_bus_id.find('\0')) << "\n";
        ss << "    Total Memory: " << format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log.print(ss.str());
    }

    // Stats and CUPTI
    auto stats = std::make_shared<rapidsmpf::Statistics>(/* enable = */ false);
#ifdef RAPIDSMPF_HAVE_CUPTI
    std::unique_ptr<rapidsmpf::CuptiMonitor> cupti_monitor;
    if (args.enable_cupti_monitoring) {
        cupti_monitor = std::make_unique<rapidsmpf::CuptiMonitor>();
        cupti_monitor->start_monitoring();
        log.print("CUPTI memory monitoring enabled");
    }
#endif

    // File selection per rank
    auto files = expand_glob(args.file_pattern);
    if (files.empty()) {
        if (comm->rank() == 0)
            log.print("No files matched pattern: " + args.file_pattern);
        if (!use_bootstrap)
            RAPIDSMPF_MPI(MPI_Finalize());
        return 1;
    }
    auto my_file = files[static_cast<std::size_t>(comm->rank()) % files.size()];
    if (comm->rank() == 0)
        log.print(
            "Using file pattern: " + args.file_pattern + ", first file: " + files.front()
        );
    log.print("Rank " + std::to_string(comm->rank()) + " reading: " + my_file);

    // Read Parquet into cudf::table
    cudf::io::parquet_reader_options reader_opts =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info{my_file});
    auto table_with_md = cudf::io::read_parquet(reader_opts);
    auto& table = table_with_md.tbl;

    // Pack per mode
    auto packed = make_packed_items(*table, args.pack_mode, stream, &br);

    // Prepare codec
    auto codec = make_codec(args.algo, args.params);

    // Runs
    std::vector<double> compress_t, decompress_t, comp_send_t, recv_decomp_t, send_t,
        recv_t;
    compress_t.reserve(args.num_runs);
    decompress_t.reserve(args.num_runs);
    comp_send_t.reserve(args.num_runs);
    recv_decomp_t.reserve(args.num_runs);
    send_t.reserve(args.num_runs);
    recv_t.reserve(args.num_runs);

    std::size_t logical_bytes = packed.total_uncompressed_bytes * args.num_ops;

    for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
        if (i == args.num_warmups + args.num_runs - 1) {
            stats = std::make_shared<rapidsmpf::Statistics>(/* enable = */ true);
        }
        auto rr = run_once(comm, args, stream, &br, stats, packed, *codec);

        double cBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.compress_s;
        double dBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.decompress_s;
        double csBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                       / rr.times.comp_send_s;
        double rdBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                       / rr.times.recv_decomp_s;
        double sBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.send_only_s;
        double rBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.recv_only_s;

        std::stringstream ss;
        ss << "compress: " << format_nbytes(cBps)
           << "/s | decompress: " << format_nbytes(dBps)
           << "/s | comp+send: " << format_nbytes(csBps)
           << "/s | recv+decomp: " << format_nbytes(rdBps)
           << "/s | send-only: " << format_nbytes(sBps)
           << "/s | recv-only: " << format_nbytes(rBps) << "/s";
        if (i < args.num_warmups)
            ss << " (warmup run)";
        log.print(ss.str());

        if (i >= args.num_warmups) {
            compress_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / cBps
            );
            decompress_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / dBps
            );
            comp_send_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / csBps
            );
            recv_decomp_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / rdBps
            );
            send_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / sBps
            );
            recv_t.push_back(
                static_cast<double>(rr.counts.logical_uncompressed_bytes) / rBps
            );
        }
    }

    // Means
    auto harmonic_mean = [](std::vector<double> const& v) {
        double denom_sum = 0.0;
        for (auto x : v)
            denom_sum += 1.0 / x;
        return static_cast<double>(v.size()) / denom_sum;
    };

    if (!compress_t.empty()) {
        double mean_elapsed_c = harmonic_mean(compress_t);
        double mean_elapsed_d = harmonic_mean(decompress_t);
        double mean_elapsed_cs = harmonic_mean(comp_send_t);
        double mean_elapsed_rd = harmonic_mean(recv_decomp_t);
        double mean_elapsed_s = harmonic_mean(send_t);
        double mean_elapsed_r = harmonic_mean(recv_t);

        std::stringstream ss;
        ss << "means: compress: " << format_nbytes(logical_bytes / mean_elapsed_c) << "/s"
           << " | decompress: " << format_nbytes(logical_bytes / mean_elapsed_d) << "/s"
           << " | comp+send: " << format_nbytes(logical_bytes / mean_elapsed_cs) << "/s"
           << " | recv+decomp: " << format_nbytes(logical_bytes / mean_elapsed_rd) << "/s"
           << " | send-only: " << format_nbytes(logical_bytes / mean_elapsed_s) << "/s"
           << " | recv-only: " << format_nbytes(logical_bytes / mean_elapsed_r) << "/s";
        log.print(ss.str());
    }

#ifdef RAPIDSMPF_HAVE_CUPTI
    if (args.enable_cupti_monitoring && cupti_monitor) {
        cupti_monitor->stop_monitoring();
        std::string csv_filename =
            args.cupti_csv_prefix + std::to_string(comm->rank()) + ".csv";
        try {
            cupti_monitor->write_csv(csv_filename);
            log.print(
                "CUPTI memory data written to " + csv_filename + " ("
                + std::to_string(cupti_monitor->get_sample_count()) + " samples, "
                + std::to_string(cupti_monitor->get_total_callback_count())
                + " callbacks)"
            );
            if (comm->rank() == 0) {
                log.print(
                    "CUPTI Callback Summary:\n" + cupti_monitor->get_callback_summary()
                );
            }
        } catch (std::exception const& e) {
            log.print("Failed to write CUPTI CSV file: " + std::string(e.what()));
        }
    }
#endif

    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Finalize());
    }
    return 0;
}
