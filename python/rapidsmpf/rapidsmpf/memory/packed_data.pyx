# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport cpp_PackedData


# Create a new PackedData from metadata and device buffers.
cdef extern from *:
    """
    #include <rapidsmpf/error.hpp>
    #include <rapidsmpf/memory/cuda_memcpy_async.hpp>

    std::unique_ptr<rapidsmpf::PackedData> cpp_packed_data_from_buffers(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        rmm::cuda_stream_view stream,
        rapidsmpf::BufferResource* br
    ) {
        return std::make_unique<rapidsmpf::PackedData>(
            std::move(metadata), br->move(std::move(gpu_data), stream)
        );
    }

    std::unique_ptr<rapidsmpf::PackedData> cpp_packed_data_from_host_bytes(
        const std::uint8_t* data,
        std::size_t size,
        rapidsmpf::BufferResource* br
    ) {
        // Minimal metadata (1 byte) to satisfy PackedData constraint
        auto metadata = std::make_unique<std::vector<std::uint8_t>>(1, 0);

        // Allocate host buffer and copy data into it
        auto reservation = br->reserve_or_fail(size, rapidsmpf::MemoryType::HOST);
        auto buffer = br->make_buffer(
            rmm::cuda_stream_default, std::move(reservation)
        );

        // Copy data into the buffer
        if (size > 0) {
            buffer->write_access([&](std::byte* dst, rmm::cuda_stream_view) {
                std::memcpy(dst, data, size);
            });
        }

        return std::make_unique<rapidsmpf::PackedData>(
            std::move(metadata), std::move(buffer)
        );
    }

    std::vector<std::uint8_t> cpp_packed_data_to_host_bytes(
        rapidsmpf::PackedData* pd
    ) {
        // Extract bytes from the data buffer (handles both host and device memory)
        auto* buf = pd->data.get();
        auto const nbytes = buf->size;
        std::vector<std::uint8_t> result(nbytes);
        if (nbytes > 0) {
            RAPIDSMPF_CUDA_TRY(rapidsmpf::cuda_memcpy_async(
                result.data(), buf->data(), nbytes, buf->stream()
            ));
            buf->stream().synchronize();
        }
        return result;
    }
    """
    unique_ptr[cpp_PackedData] cpp_packed_data_from_buffers(
        unique_ptr[vector[uint8_t]] metadata,
        unique_ptr[device_buffer] gpu_data,
        cuda_stream_view stream,
        cpp_BufferResource* br,
    ) except +ex_handler nogil

    unique_ptr[cpp_PackedData] cpp_packed_data_from_host_bytes(
        const uint8_t* data,
        size_t size,
        cpp_BufferResource* br,
    ) except + nogil

    vector[uint8_t] cpp_packed_data_to_host_bytes(
        cpp_PackedData* pd,
    ) except + nogil


cdef class PackedData:
    @staticmethod
    cdef from_librapidsmpf(unique_ptr[cpp_PackedData] obj, BufferResource br):
        cdef PackedData self = PackedData.__new__(PackedData)
        self.c_obj = move(obj)
        self._br = br
        return self

    def __init__(self):
        """Initialize an empty PackedData instance."""
        pass

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    @classmethod
    def from_host_bytes(
        cls, const uint8_t[::1] data not None, BufferResource br not None
    ):
        """
        Construct a PackedData from raw host bytes.

        The bytes are stored in the data buffer (as host memory) with minimal
        metadata. This is useful for scalar allreduce operations.

        Note: This makes a copy of the input data.

        Parameters
        ----------
        data
            Contiguous buffer of bytes (bytes, bytearray, or buffer-protocol object).
        br
            Buffer resource for memory allocation.

        Returns
        -------
        A new PackedData instance containing the bytes.
        """
        cdef cpp_BufferResource* _br = br.ptr()
        cdef PackedData ret = cls.__new__(cls)
        cdef size_t size = len(data)
        cdef const uint8_t* data_ptr = NULL
        if size > 0:
            data_ptr = <const uint8_t*>&data[0]
        with nogil:
            ret.c_obj = cpp_packed_data_from_host_bytes(data_ptr, size, _br)
        ret._br = br
        return ret

    def to_host_bytes(self) -> bytes:
        """
        Extract the host bytes from this PackedData.

        Returns the bytes stored in the data buffer. Works with both
        host and device memory buffers.

        Returns a copy of the bytes stored in the data buffer. Works with both
        host and device memory buffers. The method synchronizes with the
        underlying buffer's CUDA stream before returning.

        Returns
        -------
        The raw bytes.

        Raises
        ------
        ValueError
            If the PackedData is empty.
        """
        if not self.c_obj:
            raise ValueError("PackedData is empty")

        cdef vector[uint8_t] result
        with nogil:
            result = cpp_packed_data_to_host_bytes(self.c_obj.get())
        return bytes(result)


# Convert a vector of `cpp_PackedData` into a list of `PackedData`.
cdef list packed_data_vector_to_list(
    vector[cpp_PackedData] packed_data, BufferResource br
):
    cdef list ret = []
    for i in range(packed_data.size()):
        item = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(packed_data[i])),
            br,
        )
        ret.append(item)
    return ret
