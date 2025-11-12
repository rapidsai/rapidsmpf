/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include <rapidsmpf/nvcomp.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>

#include <nvcomp/cascaded.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>

namespace rapidsmpf {

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
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override {
        nvcompBatchedLZ4CompressOpts_t copts = nvcompBatchedLZ4CompressDefaultOpts;
        nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
        nvcomp::LZ4Manager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        if (br != nullptr) {
            auto reservation = br->reserve_or_fail(sizeof(size_t), MemoryType::DEVICE);
            auto size_buf = br->allocate(sizeof(size_t), stream, reservation);
            size_buf->write_access([&](std::byte* sz_ptr, rmm::cuda_stream_view s) {
                mgr.compress(
                    static_cast<uint8_t const*>(d_in),
                    static_cast<uint8_t*>(d_out),
                    cfg,
                    reinterpret_cast<size_t*>(sz_ptr)
                );
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    out_bytes, sz_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, s
                ));
            });
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        } else {
            size_t* d_size = nullptr;
            RAPIDSMPF_CUDA_TRY(cudaMallocAsync(
                reinterpret_cast<void**>(&d_size), sizeof(size_t), stream.value()
            ));
            mgr.compress(
                static_cast<uint8_t const*>(d_in),
                static_cast<uint8_t*>(d_out),
                cfg,
                d_size
            );
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                out_bytes, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream.value()
            ));
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
            RAPIDSMPF_CUDA_TRY(cudaFreeAsync(d_size, stream.value()));
        }
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
    CascadedCodec(std::size_t chunk_size, int rle, int delta, bool bitpack)
        : chunk_size_{chunk_size} {
        copts_ = nvcompBatchedCascadedCompressDefaultOpts;
        copts_.num_RLEs = rle;
        copts_.num_deltas = delta;
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
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override {
        nvcomp::CascadedManager mgr{chunk_size_, copts_, dopts_, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        if (br != nullptr) {
            auto reservation = br->reserve_or_fail(sizeof(size_t), MemoryType::DEVICE);
            auto size_buf = br->allocate(sizeof(size_t), stream, reservation);
            size_buf->write_access([&](std::byte* sz_ptr, rmm::cuda_stream_view s) {
                mgr.compress(
                    static_cast<uint8_t const*>(d_in),
                    static_cast<uint8_t*>(d_out),
                    cfg,
                    reinterpret_cast<size_t*>(sz_ptr)
                );
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    out_bytes, sz_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, s
                ));
            });
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        } else {
            size_t* d_size = nullptr;
            RAPIDSMPF_CUDA_TRY(cudaMallocAsync(
                reinterpret_cast<void**>(&d_size), sizeof(size_t), stream.value()
            ));
            mgr.compress(
                static_cast<uint8_t const*>(d_in),
                static_cast<uint8_t*>(d_out),
                cfg,
                d_size
            );
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                out_bytes, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream.value()
            ));
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
            RAPIDSMPF_CUDA_TRY(cudaFreeAsync(d_size, stream.value()));
        }
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

class SnappyCodec final : public NvcompCodec {
  public:
    explicit SnappyCodec(std::size_t chunk_size) : chunk_size_{chunk_size} {}

    std::size_t get_max_compressed_bytes(
        std::size_t in_bytes, rmm::cuda_stream_view stream
    ) override {
        nvcompBatchedSnappyCompressOpts_t copts = nvcompBatchedSnappyCompressDefaultOpts;
        nvcompBatchedSnappyDecompressOpts_t dopts =
            nvcompBatchedSnappyDecompressDefaultOpts;
        nvcomp::SnappyManager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        return cfg.max_compressed_buffer_size;
    }

    void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override {
        nvcompBatchedSnappyCompressOpts_t copts = nvcompBatchedSnappyCompressDefaultOpts;
        nvcompBatchedSnappyDecompressOpts_t dopts =
            nvcompBatchedSnappyDecompressDefaultOpts;
        nvcomp::SnappyManager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        if (br != nullptr) {
            auto reservation = br->reserve_or_fail(sizeof(size_t), MemoryType::DEVICE);
            auto size_buf = br->allocate(sizeof(size_t), stream, reservation);
            size_buf->write_access([&](std::byte* sz_ptr, rmm::cuda_stream_view s) {
                mgr.compress(
                    static_cast<uint8_t const*>(d_in),
                    static_cast<uint8_t*>(d_out),
                    cfg,
                    reinterpret_cast<size_t*>(sz_ptr)
                );
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    out_bytes, sz_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, s
                ));
            });
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        } else {
            size_t* d_size = nullptr;
            RAPIDSMPF_CUDA_TRY(cudaMallocAsync(
                reinterpret_cast<void**>(&d_size), sizeof(size_t), stream.value()
            ));
            mgr.compress(
                static_cast<uint8_t const*>(d_in),
                static_cast<uint8_t*>(d_out),
                cfg,
                d_size
            );
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                out_bytes, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream.value()
            ));
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
            RAPIDSMPF_CUDA_TRY(cudaFreeAsync(d_size, stream.value()));
        }
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        (void)out_bytes;
        nvcompBatchedSnappyCompressOpts_t copts = nvcompBatchedSnappyCompressDefaultOpts;
        nvcompBatchedSnappyDecompressOpts_t dopts =
            nvcompBatchedSnappyDecompressDefaultOpts;
        nvcomp::SnappyManager mgr{chunk_size_, copts, dopts, stream.value()};
        const uint8_t* in_ptrs[1] = {static_cast<uint8_t const*>(d_in)};
        size_t in_sizes[1] = {in_bytes};
        auto cfgs = mgr.configure_decompression(in_ptrs, 1, in_sizes);
        uint8_t* out_ptrs[1] = {static_cast<uint8_t*>(d_out)};
        mgr.decompress(out_ptrs, in_ptrs, cfgs, nullptr);
    }

  private:
    std::size_t chunk_size_;
};

class ZstdCodec final : public NvcompCodec {
  public:
    explicit ZstdCodec(std::size_t chunk_size) : chunk_size_{chunk_size} {}

    std::size_t get_max_compressed_bytes(
        std::size_t in_bytes, rmm::cuda_stream_view stream
    ) override {
        nvcompBatchedZstdCompressOpts_t copts = nvcompBatchedZstdCompressDefaultOpts;
        nvcompBatchedZstdDecompressOpts_t dopts = nvcompBatchedZstdDecompressDefaultOpts;
        nvcomp::ZstdManager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        return cfg.max_compressed_buffer_size;
    }

    void compress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t* out_bytes,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override {
        nvcompBatchedZstdCompressOpts_t copts = nvcompBatchedZstdCompressDefaultOpts;
        nvcompBatchedZstdDecompressOpts_t dopts = nvcompBatchedZstdDecompressDefaultOpts;
        nvcomp::ZstdManager mgr{chunk_size_, copts, dopts, stream.value()};
        auto cfg = mgr.configure_compression(in_bytes);
        if (br != nullptr) {
            auto reservation = br->reserve_or_fail(sizeof(size_t), MemoryType::DEVICE);
            auto size_buf = br->allocate(sizeof(size_t), stream, reservation);
            size_buf->write_access([&](std::byte* sz_ptr, rmm::cuda_stream_view s) {
                mgr.compress(
                    static_cast<uint8_t const*>(d_in),
                    static_cast<uint8_t*>(d_out),
                    cfg,
                    reinterpret_cast<size_t*>(sz_ptr)
                );
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    out_bytes, sz_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, s
                ));
            });
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        } else {
            size_t* d_size = nullptr;
            RAPIDSMPF_CUDA_TRY(cudaMallocAsync(
                reinterpret_cast<void**>(&d_size), sizeof(size_t), stream.value()
            ));
            mgr.compress(
                static_cast<uint8_t const*>(d_in),
                static_cast<uint8_t*>(d_out),
                cfg,
                d_size
            );
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                out_bytes, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream.value()
            ));
            RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
            RAPIDSMPF_CUDA_TRY(cudaFreeAsync(d_size, stream.value()));
        }
    }

    void decompress(
        void const* d_in,
        std::size_t in_bytes,
        void* d_out,
        std::size_t out_bytes,
        rmm::cuda_stream_view stream
    ) override {
        (void)out_bytes;
        nvcompBatchedZstdCompressOpts_t copts = nvcompBatchedZstdCompressDefaultOpts;
        nvcompBatchedZstdDecompressOpts_t dopts = nvcompBatchedZstdDecompressDefaultOpts;
        nvcomp::ZstdManager mgr{chunk_size_, copts, dopts, stream.value()};
        const uint8_t* in_ptrs[1] = {static_cast<uint8_t const*>(d_in)};
        size_t in_sizes[1] = {in_bytes};
        auto cfgs = mgr.configure_decompression(in_ptrs, 1, in_sizes);
        uint8_t* out_ptrs[1] = {static_cast<uint8_t*>(d_out)};
        mgr.decompress(out_ptrs, in_ptrs, cfgs, nullptr);
    }

  private:
    std::size_t chunk_size_;
};

std::unique_ptr<NvcompCodec> make_codec(Algo algo, KvParams const& p) {
    switch (algo) {
    case Algo::LZ4:
        return std::make_unique<LZ4Codec>(p.chunk_size);
    case Algo::Zstd:
        return std::make_unique<ZstdCodec>(p.chunk_size);
    case Algo::Snappy:
        return std::make_unique<SnappyCodec>(p.chunk_size);
    case Algo::Cascaded:
    default:
        return std::make_unique<CascadedCodec>(
            p.chunk_size, p.cascaded_rle, p.cascaded_delta, p.cascaded_bitpack
        );
    }
}

}  // namespace rapidsmpf
