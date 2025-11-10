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
    Column
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
    virtual std::size_t get_max_compressed_bytes(
        std::size_t uncompressed_bytes, rmm::cuda_stream_view stream
    ) = 0;
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

    std::size_t get_max_compressed_bytes(
        std::size_t in_bytes, rmm::cuda_stream_view stream
    ) override {
        nvcompBatchedLZ4CompressOpts_t copts = nvcompBatchedLZ4CompressDefaultOpts;
        nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
        nvcomp::LZ4Manager mgr{chunk_size_, copts, dopts, stream.value()};
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
        nvcompBatchedLZ4CompressOpts_t copts = nvcompBatchedLZ4CompressDefaultOpts;
        nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
        nvcomp::LZ4Manager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        size_t* pinned_bytes = nullptr;
        RAPIDSMPF_CUDA_TRY(cudaHostAlloc(
            reinterpret_cast<void**>(&pinned_bytes), sizeof(size_t), cudaHostAllocDefault
        ));
        mgr.compress(
            static_cast<uint8_t const*>(d_in),
            static_cast<uint8_t*>(d_out),
            cfg,
            pinned_bytes
        );
        RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        *out_bytes = *pinned_bytes;
        RAPIDSMPF_CUDA_TRY(cudaFreeHost(pinned_bytes));
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        (void)out_bytes;
        nvcompBatchedLZ4CompressOpts_t copts = nvcompBatchedLZ4CompressDefaultOpts;
        nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
        nvcomp::LZ4Manager mgr{chunk_size_, copts, dopts, stream.value()};
        const uint8_t* in_ptrs[1] = {static_cast<uint8_t const*>(d_in)};
        size_t in_sizes[1] = {in_bytes};
        auto cfgs = mgr.configure_decompression(in_ptrs, 1, in_sizes);
        uint8_t* out_ptrs[1] = {static_cast<uint8_t*>(d_out)};
        mgr.decompress(out_ptrs, in_ptrs, cfgs, nullptr);
    }

  private:
    std::size_t chunk_size_;
};

class CascadedCodec final : public NvcompCodec {
  public:
    CascadedCodec(std::size_t chunk_size, int rle, int delta, int bitpack)
        : chunk_size_{chunk_size} {
        copts_ = nvcompBatchedCascadedCompressDefaultOpts;
        copts_.num_RLEs = rle ? 1 : 0;
        copts_.num_deltas = delta ? 1 : 0;
        copts_.use_bp = bitpack ? 1 : 0;
        dopts_ = nvcompBatchedCascadedDecompressDefaultOpts;
    }

    std::size_t get_max_compressed_bytes(
        std::size_t in_bytes, rmm::cuda_stream_view stream
    ) override {
        nvcomp::CascadedManager mgr{chunk_size_, copts_, dopts_, stream.value()};
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
        nvcomp::CascadedManager mgr{chunk_size_, copts_, dopts_, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        size_t* pinned_bytes = nullptr;
        RAPIDSMPF_CUDA_TRY(cudaHostAlloc(
            reinterpret_cast<void**>(&pinned_bytes), sizeof(size_t), cudaHostAllocDefault
        ));
        mgr.compress(
            static_cast<uint8_t const*>(d_in),
            static_cast<uint8_t*>(d_out),
            cfg,
            pinned_bytes
        );
        RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        *out_bytes = *pinned_bytes;
        RAPIDSMPF_CUDA_TRY(cudaFreeHost(pinned_bytes));
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        (void)out_bytes;
        nvcomp::CascadedManager mgr{chunk_size_, copts_, dopts_, stream.value()};
        const uint8_t* in_ptrs[1] = {static_cast<uint8_t const*>(d_in)};
        size_t in_sizes[1] = {in_bytes};
        auto cfgs = mgr.configure_decompression(in_ptrs, 1, in_sizes);
        uint8_t* out_ptrs[1] = {static_cast<uint8_t*>(d_out)};
        mgr.decompress(out_ptrs, in_ptrs, cfgs, nullptr);
    }

  private:
    std::size_t chunk_size_{};
    nvcompBatchedCascadedCompressOpts_t copts_{};
    nvcompBatchedCascadedDecompressOpts_t dopts_{};
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
    return std::make_unique<PackedData>(
        std::move(packed.metadata), br->move(std::move(packed.gpu_data), stream)
    );
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
                        else if (v == "column")
                            args_.pack_mode = PackMode::Column;
                        else
                            RAPIDSMPF_FAIL(
                                "-P must be one of {table, column}", std::invalid_argument
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
                           << "  -P <mode>    Packing mode {table, column} (default: "
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
    // Uncompressed bytes that represent the actual payload we transmit (device data only)
    std::size_t total_uncompressed_bytes{0};
    // Convenience: device payload bytes (same as total_uncompressed_bytes here)
    std::size_t total_payload_bytes{0};
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
        ret.total_uncompressed_bytes += item.packed->data->size;
        ret.total_payload_bytes += item.packed->data->size;
        ret.items.emplace_back(std::move(item));
    } else {
        auto tv = table.view();
        for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
            cudf::table_view col_tv{std::vector<cudf::column_view>{tv.column(i)}};
            auto item = PackedItem{};
            item.packed = pack_table_to_packed(col_tv, stream, br);
            ret.total_uncompressed_bytes += item.packed->data->size;
            ret.total_payload_bytes += item.packed->data->size;
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
    // Round-trip totals measured at initiator
    double rt_nocomp_s{0.0};
    double rt_comp_s{0.0};
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
    NvcompCodec& codec,
    std::uint64_t run_index
) {
    (void)statistics;
    auto const nranks = comm->nranks();
    auto const rank = comm->rank();
    auto const dst = static_cast<Rank>((rank + 1) % nranks);
    auto const src = static_cast<Rank>((rank - 1 + nranks) % nranks);

    Tag tag_ping_nc{10, 0};
    Tag tag_pong_nc{10, 1};
    Tag tag_ping_c{11, 0};
    Tag tag_pong_c{11, 1};

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
        std::size_t const max_out =
            (in_bytes == 0) ? 1 : codec.get_max_compressed_bytes(in_bytes, stream);
        auto reservation = br->reserve_or_fail(max_out, MemoryType::DEVICE);
        comp_outputs.emplace_back(br->allocate(max_out, stream, reservation));
    }

    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto t0 = Clock::now();
    // Compress all items (single batch) on stream
    for (std::size_t i = 0; i < data.items.size(); ++i) {
        auto const in_bytes = data.items[i].packed->data->size;
        if (in_bytes == 0) {
            comp_output_sizes[i] = 0;
            continue;
        }
        // Ensure any prior writes to input are completed
        data.items[i].packed->data->stream().synchronize();
        // Launch compression on the output buffer's stream and record an event after
        comp_outputs[i]->write_access(
            [&codec, &data, i, in_bytes, &comp_output_sizes, stream](
                std::byte* out_ptr, rmm::cuda_stream_view out_stream
            ) {
                (void)out_ptr;  // pointer used below
                // Lock input for raw pointer access
                auto* in_raw = data.items[i].packed->data->exclusive_data_access();
                std::size_t out_bytes = 0;
                codec.compress(
                    static_cast<void const*>(in_raw),
                    in_bytes,
                    static_cast<void*>(out_ptr),
                    &out_bytes,
                    out_stream
                );
                // Ensure comp_bytes is populated before returning
                RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(out_stream.value()));
                data.items[i].packed->data->unlock();
                comp_output_sizes[i] = out_bytes;
            }
        );
    }
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto t1 = Clock::now();

    // Phase A (RTT no compression): ping-pong per op (sequential per item to avoid
    // deadlocks)
    Duration rt_nc_total{0};
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        bool initiator =
            ((static_cast<std::uint64_t>(rank) + op + run_index) % 2ull) == 0ull;
        auto rt_start = Clock::now();
        if (initiator) {
            for (std::size_t i = 0; i < nocomp_payloads.size(); ++i) {
                // post pong recv and send ping, then wait both
                auto res_r =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto recv_buf = br->allocate(nocomp_payloads[i]->size, stream, res_r);
                std::vector<std::unique_ptr<Communicator::Future>> futs;
                futs.push_back(comm->recv(dst, tag_pong_nc, std::move(recv_buf)));
                auto res_s =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto send_buf = br->allocate(nocomp_payloads[i]->size, stream, res_s);
                buffer_copy(*send_buf, *nocomp_payloads[i], nocomp_payloads[i]->size);
                if (!send_buf->is_latest_write_done())
                    send_buf->stream().synchronize();
                futs.push_back(comm->send(std::move(send_buf), dst, tag_ping_nc));
                while (!futs.empty()) {
                    std::ignore = comm->test_some(futs);
                }
            }
        } else {
            // Responder: for each item, recv ping then send pong
            for (std::size_t i = 0; i < nocomp_payloads.size(); ++i) {
                auto res_r =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto recv_buf = br->allocate(nocomp_payloads[i]->size, stream, res_r);
                std::vector<std::unique_ptr<Communicator::Future>> rf;
                rf.push_back(comm->recv(src, tag_ping_nc, std::move(recv_buf)));
                while (!rf.empty()) {
                    std::ignore = comm->test_some(rf);
                }
                auto res_s =
                    br->reserve_or_fail(nocomp_payloads[i]->size, MemoryType::DEVICE);
                auto send_buf = br->allocate(nocomp_payloads[i]->size, stream, res_s);
                buffer_copy(*send_buf, *nocomp_payloads[i], nocomp_payloads[i]->size);
                if (!send_buf->is_latest_write_done())
                    send_buf->stream().synchronize();
                std::vector<std::unique_ptr<Communicator::Future>> sf;
                sf.push_back(comm->send(std::move(send_buf), src, tag_pong_nc));
                while (!sf.empty()) {
                    std::ignore = comm->test_some(sf);
                }
            }
        }
        auto rt_end = Clock::now();
        // Each rank measures its own RTT locally
        rt_nc_total += (rt_end - rt_start);
    }

    // Phase B (RTT compressed payload only): ping-pong with size headers per item
    Duration rt_c_total{0};
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        bool initiator =
            ((static_cast<std::uint64_t>(rank) + op + run_index) % 2ull) == 0ull;
        auto rt_start = Clock::now();
        if (initiator) {
            for (std::size_t i = 0; i < data.items.size(); ++i) {
                // Send header with size to dst
                std::uint64_t sz = static_cast<std::uint64_t>(comp_output_sizes[i]);
                auto res_h = br->reserve_or_fail(sizeof(std::uint64_t), MemoryType::HOST);
                auto hdr = br->allocate(sizeof(std::uint64_t), stream, res_h);
                hdr->write_access([&](std::byte* p, rmm::cuda_stream_view) {
                    std::memcpy(p, &sz, sizeof(std::uint64_t));
                });
                if (!hdr->is_latest_write_done())
                    hdr->stream().synchronize();
                std::vector<std::unique_ptr<Communicator::Future>> hf;
                hf.push_back(comm->send(std::move(hdr), dst, tag_ping_c));
                while (!hf.empty()) {
                    std::ignore = comm->test_some(hf);
                }
                // Receive pong header with size from src (blocking wait)
                auto res_hr =
                    br->reserve_or_fail(sizeof(std::uint64_t), MemoryType::HOST);
                auto hdr_r = br->allocate(sizeof(std::uint64_t), stream, res_hr);
                auto fut_hdr = comm->recv(dst, tag_pong_c, std::move(hdr_r));
                auto hdr_buf = comm->wait(std::move(fut_hdr));
                auto* p = hdr_buf->exclusive_data_access();
                std::uint64_t pong_sz = 0;
                std::memcpy(&pong_sz, p, sizeof(std::uint64_t));
                hdr_buf->unlock();
                // Send ping payload (if any)
                if (sz > 0) {
                    auto res_s = br->reserve_or_fail(sz, MemoryType::DEVICE);
                    auto send_buf = br->allocate(sz, stream, res_s);
                    if (comp_output_sizes[i] > 0) {
                        buffer_copy(*send_buf, *comp_outputs[i], comp_output_sizes[i]);
                    }
                    if (!send_buf->is_latest_write_done())
                        send_buf->stream().synchronize();
                    std::vector<std::unique_ptr<Communicator::Future>> sf;
                    sf.push_back(comm->send(std::move(send_buf), dst, tag_ping_c));
                    while (!sf.empty()) {
                        std::ignore = comm->test_some(sf);
                    }
                }
                // Receive pong payload of announced size
                if (pong_sz > 0) {
                    auto res_r = br->reserve_or_fail(pong_sz, MemoryType::DEVICE);
                    auto recv_buf = br->allocate(pong_sz, stream, res_r);
                    std::vector<std::unique_ptr<Communicator::Future>> rf;
                    rf.push_back(comm->recv(dst, tag_pong_c, std::move(recv_buf)));
                    while (!rf.empty()) {
                        std::ignore = comm->test_some(rf);
                    }
                }
            }
        } else {
            for (std::size_t i = 0; i < data.items.size(); ++i) {
                // Receive ping header with size (blocking wait)
                auto res_hr =
                    br->reserve_or_fail(sizeof(std::uint64_t), MemoryType::HOST);
                auto hdr_r = br->allocate(sizeof(std::uint64_t), stream, res_hr);
                auto fut_hdr = comm->recv(src, tag_ping_c, std::move(hdr_r));
                auto hdr_buf = comm->wait(std::move(fut_hdr));
                auto* p = hdr_buf->exclusive_data_access();
                std::uint64_t ping_sz = 0;
                std::memcpy(&ping_sz, p, sizeof(std::uint64_t));
                hdr_buf->unlock();
                // Send pong header with our size
                std::uint64_t sz = static_cast<std::uint64_t>(comp_output_sizes[i]);
                auto res_h = br->reserve_or_fail(sizeof(std::uint64_t), MemoryType::HOST);
                auto hdr = br->allocate(sizeof(std::uint64_t), stream, res_h);
                hdr->write_access([&](std::byte* q, rmm::cuda_stream_view) {
                    std::memcpy(q, &sz, sizeof(std::uint64_t));
                });
                if (!hdr->is_latest_write_done())
                    hdr->stream().synchronize();
                std::vector<std::unique_ptr<Communicator::Future>> hf;
                hf.push_back(comm->send(std::move(hdr), src, tag_pong_c));
                while (!hf.empty()) {
                    std::ignore = comm->test_some(hf);
                }
                // Receive ping payload
                if (ping_sz > 0) {
                    auto res_r = br->reserve_or_fail(ping_sz, MemoryType::DEVICE);
                    auto recv_buf = br->allocate(ping_sz, stream, res_r);
                    std::vector<std::unique_ptr<Communicator::Future>> rf;
                    rf.push_back(comm->recv(src, tag_ping_c, std::move(recv_buf)));
                    while (!rf.empty()) {
                        std::ignore = comm->test_some(rf);
                    }
                }
                // Send pong payload
                if (sz > 0) {
                    auto res_s = br->reserve_or_fail(sz, MemoryType::DEVICE);
                    auto send_buf = br->allocate(sz, stream, res_s);
                    if (comp_output_sizes[i] > 0) {
                        buffer_copy(*send_buf, *comp_outputs[i], comp_output_sizes[i]);
                    }
                    if (!send_buf->is_latest_write_done())
                        send_buf->stream().synchronize();
                    std::vector<std::unique_ptr<Communicator::Future>> sf;
                    sf.push_back(comm->send(std::move(send_buf), src, tag_pong_c));
                    while (!sf.empty()) {
                        std::ignore = comm->test_some(sf);
                    }
                }
            }
        }
        auto rt_end = Clock::now();
        rt_c_total += (rt_end - rt_start);
    }

    // Decompress received buffers (simulate by decompressing our own produced outputs in
    // symmetric setup)
    auto c0 = Clock::now();
    for (std::size_t i = 0; i < data.items.size(); ++i) {
        auto const out_bytes = data.items[i].packed->data->size;
        if (out_bytes == 0) {
            continue;
        }
        auto res = br->reserve_or_fail(out_bytes, MemoryType::DEVICE);
        auto out = br->allocate(out_bytes, stream, res);
        // Ensure compressed outputs are ready before using as input
        comp_outputs[i]->stream().synchronize();
        out->write_access([&codec, &comp_outputs, &comp_output_sizes, i, out_bytes](
                              std::byte* out_ptr, rmm::cuda_stream_view out_stream
                          ) {
            auto* in_raw = comp_outputs[i]->exclusive_data_access();
            codec.decompress(
                static_cast<void const*>(in_raw),
                comp_output_sizes[i],
                static_cast<void*>(out_ptr),
                out_bytes,
                out_stream
            );
            comp_outputs[i]->unlock();
        });
    }
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    auto c1 = Clock::now();

    RunResult result{};
    result.times.compress_s = std::chrono::duration<double>(t1 - t0).count();
    result.times.rt_nocomp_s = rt_nc_total.count();
    result.times.rt_comp_s = rt_c_total.count();
    result.times.decompress_s = std::chrono::duration<double>(c1 - c0).count();

    // Use payload (device) bytes as the logical uncompressed size for throughput
    result.counts.logical_uncompressed_bytes = data.total_payload_bytes * args.num_ops;
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
    std::vector<double> compress_t, decompress_t, rt_nc_t, rt_c_t;
    compress_t.reserve(args.num_runs);
    decompress_t.reserve(args.num_runs);
    rt_nc_t.reserve(args.num_runs);
    rt_c_t.reserve(args.num_runs);

    std::size_t logical_bytes = packed.total_uncompressed_bytes * args.num_ops;

    for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
        if (i == args.num_warmups + args.num_runs - 1) {
            stats = std::make_shared<rapidsmpf::Statistics>(/* enable = */ true);
        }
        auto rr = run_once(comm, args, stream, &br, stats, packed, *codec, i);

        double cBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.compress_s;
        double dBps = static_cast<double>(rr.counts.logical_uncompressed_bytes)
                      / rr.times.decompress_s;
        // Round-trip one-way throughput: 2 * bytes_one_way / RTT
        double rt_nc_Bps =
            rr.times.rt_nocomp_s > 0.0
                ? (2.0 * static_cast<double>(rr.counts.logical_uncompressed_bytes))
                      / rr.times.rt_nocomp_s
                : 0.0;
        double rt_c_Bps =
            rr.times.rt_comp_s > 0.0
                ? (2.0 * static_cast<double>(rr.counts.logical_uncompressed_bytes))
                      / rr.times.rt_comp_s
                : 0.0;

        std::stringstream ss;
        ss << "compress: " << format_nbytes(cBps)
           << "/s | decompress: " << format_nbytes(dBps)
           << "/s | rt(nocomp): " << format_nbytes(rt_nc_Bps)
           << "/s | rt(comp): " << format_nbytes(rt_c_Bps) << "/s";
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
            rt_nc_t.push_back(rr.times.rt_nocomp_s);
            rt_c_t.push_back(rr.times.rt_comp_s);
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
        double mean_rt_nc = harmonic_mean(rt_nc_t);
        double mean_rt_c = harmonic_mean(rt_c_t);

        std::stringstream ss;
        ss << "means: compress: " << format_nbytes(logical_bytes / mean_elapsed_c) << "/s"
           << " | decompress: " << format_nbytes(logical_bytes / mean_elapsed_d) << "/s"
           << " | rt(nocomp): "
           << format_nbytes((2.0 * static_cast<double>(logical_bytes)) / mean_rt_nc)
           << "/s | rt(comp): "
           << format_nbytes((2.0 * static_cast<double>(logical_bytes)) / mean_rt_c)
           << "/s";
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
