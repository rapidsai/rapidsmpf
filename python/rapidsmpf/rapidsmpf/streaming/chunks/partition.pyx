# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.packed_data cimport PackedData, cpp_PackedData
from rapidsmpf.streaming.core.message cimport Message, cpp_Message


cdef extern from "<rapidsmpf/streaming/chunks/partition.hpp>" nogil:
    cpp_Message cpp_to_message"rapidsmpf::streaming::to_message"\
        (uint64_t sequence_number, unique_ptr[cpp_PartitionMapChunk]) \
        except +ex_handler
    cpp_Message cpp_to_message"rapidsmpf::streaming::to_message"\
        (uint64_t sequence_number, unique_ptr[cpp_PartitionVectorChunk]) \
        except +ex_handler


# Move PackedData into a chunk's container. We implement these in C++ because
# PackedData doesn't have a default ctor.
cdef extern from * nogil:
    """
    namespace {
    void cpp_insert_into_partition_map(
        rapidsmpf::streaming::PartitionMapChunk* chunk,
        std::uint32_t pid,
        std::unique_ptr<rapidsmpf::PackedData> packed_data
    ) {
        chunk->data.emplace(pid, std::move(*packed_data));
    }

    void cpp_append_to_partition_vector(
        rapidsmpf::streaming::PartitionVectorChunk* chunk,
        std::unique_ptr<rapidsmpf::PackedData> packed_data
    ) {
        chunk->data.push_back(std::move(*packed_data));
    }

    void cpp_drain_partition_map(
        rapidsmpf::streaming::PartitionMapChunk* chunk,
        std::vector<std::uint32_t>& keys,
        std::vector<std::unique_ptr<rapidsmpf::PackedData>>& values
    ) {
        keys.reserve(chunk->data.size());
        values.reserve(chunk->data.size());
        for (auto& [pid, pd] : chunk->data) {
            keys.push_back(pid);
            values.push_back(std::make_unique<rapidsmpf::PackedData>(std::move(pd)));
        }
        chunk->data.clear();
    }

    void cpp_drain_partition_vector(
        rapidsmpf::streaming::PartitionVectorChunk* chunk,
        std::vector<std::unique_ptr<rapidsmpf::PackedData>>& values
    ) {
        values.reserve(chunk->data.size());
        for (auto& pd : chunk->data) {
            values.push_back(std::make_unique<rapidsmpf::PackedData>(std::move(pd)));
        }
        chunk->data.clear();
    }
    }  // namespace
    """
    void cpp_insert_into_partition_map(
        cpp_PartitionMapChunk* chunk,
        uint32_t pid,
        unique_ptr[cpp_PackedData] packed_data,
    ) except +ex_handler
    void cpp_append_to_partition_vector(
        cpp_PartitionVectorChunk* chunk,
        unique_ptr[cpp_PackedData] packed_data,
    ) except +ex_handler
    void cpp_drain_partition_map(
        cpp_PartitionMapChunk* chunk,
        vector[uint32_t]& keys,
        vector[unique_ptr[cpp_PackedData]]& values,
    ) except +ex_handler
    void cpp_drain_partition_vector(
        cpp_PartitionVectorChunk* chunk,
        vector[unique_ptr[cpp_PackedData]]& values,
    ) except +ex_handler


cdef class PartitionMapChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    def from_packed_data_map(dict data not None, BufferResource br not None):
        """
        Construct a PartitionMapChunk from a mapping of partition ID to PackedData.

        Parameters
        ----------
        data
            Mapping of partition ID to the
            :class:`~rapidsmpf.memory.packed_data.PackedData` it holds. Each PackedData
            is consumed and left empty after this call.
        br
            Buffer resource kept alive for the lifetime of the chunk.

        Returns
        -------
        A new PartitionMapChunk owning the given packed data.

        Raises
        ------
        ValueError
            If any of the provided PackedData objects is empty.
        """
        cdef unique_ptr[cpp_PartitionMapChunk] handle = make_unique[
            cpp_PartitionMapChunk
        ]()
        cdef uint32_t pid
        cdef PackedData pd
        for key, value in data.items():
            pid = key
            pd = <PackedData?>value
            if not pd.c_obj:
                raise ValueError("PackedData was empty")
            cpp_insert_into_partition_map(handle.get(), pid, move(pd.c_obj))
        return PartitionMapChunk.from_handle(move(handle), br)

    def to_packed_data_map(self):
        """
        Extract the partition data as a mapping of partition ID to PackedData.

        The chunk is drained and left empty after this call.

        Returns
        -------
        A dict mapping partition ID to the
        :class:`~rapidsmpf.memory.packed_data.PackedData` it holds.
        """
        cdef vector[uint32_t] keys
        cdef vector[unique_ptr[cpp_PackedData]] values
        cpp_drain_partition_map(self._handle.get(), keys, values)
        cdef dict ret = {}
        cdef size_t i
        for i in range(values.size()):
            ret[keys[i]] = PackedData.from_librapidsmpf(move(values[i]), self._br)
        return ret

    @staticmethod
    cdef PartitionMapChunk from_handle(
        unique_ptr[cpp_PartitionMapChunk] handle, BufferResource br
    ):
        """
        Construct a PartitionMapChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PartitionMapChunk.

        Returns
        -------
        A new PartitionMapChunk wrapping the given handle.
        """

        cdef PartitionMapChunk ret = PartitionMapChunk.__new__(PartitionMapChunk)
        ret._handle = move(handle)
        ret._br = br
        return ret

    @staticmethod
    def from_message(Message message not None, BufferResource br not None):
        """
        Construct a PartitionMapChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PartitionMapChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new PartitionMapChunk extracted from the given message.
        """
        return PartitionMapChunk.from_handle(
            make_unique[cpp_PartitionMapChunk](
                message._handle.release[cpp_PartitionMapChunk]()
            ),
            br,
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this PartitionMapChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PartitionMapChunk
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this PartitionMapChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PartitionMapChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_PartitionMapChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PartitionMapChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionMapChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PartitionMapChunk] release_handle(self):
        """
        Release ownership of the underlying C++ PartitionMapChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionMapChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)


cdef class PartitionVectorChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    def from_packed_data_list(list data not None, BufferResource br not None):
        """
        Construct a PartitionVectorChunk from a sequence of PackedData.

        Parameters
        ----------
        data
            Sequence of :class:`~rapidsmpf.memory.packed_data.PackedData` objects,
            stored in order. Each PackedData is consumed and left empty after this
            call.
        br
            Buffer resource kept alive for the lifetime of the chunk.

        Returns
        -------
        A new PartitionVectorChunk owning the given packed data.

        Raises
        ------
        ValueError
            If any of the provided PackedData objects is empty.
        """
        cdef unique_ptr[cpp_PartitionVectorChunk] handle = make_unique[
            cpp_PartitionVectorChunk
        ]()
        cdef PackedData pd
        for value in data:
            pd = <PackedData?>value
            if not pd.c_obj:
                raise ValueError("PackedData was empty")
            cpp_append_to_partition_vector(handle.get(), move(pd.c_obj))
        return PartitionVectorChunk.from_handle(move(handle), br)

    def to_packed_data_list(self):
        """
        Extract the partition data as a list of PackedData.

        The chunk is drained and left empty after this call.

        Returns
        -------
        A list of :class:`~rapidsmpf.memory.packed_data.PackedData`, in order.
        """
        cdef vector[unique_ptr[cpp_PackedData]] values
        cpp_drain_partition_vector(self._handle.get(), values)
        cdef list ret = []
        cdef size_t i
        for i in range(values.size()):
            ret.append(PackedData.from_librapidsmpf(move(values[i]), self._br))
        return ret

    @staticmethod
    cdef PartitionVectorChunk from_handle(
        unique_ptr[cpp_PartitionVectorChunk] handle, BufferResource br
    ):
        """
        Construct a PartitionVectorChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PartitionVectorChunk.

        Returns
        -------
        A new PartitionVectorChunk wrapping the given handle.
        """
        cdef PartitionVectorChunk ret = PartitionVectorChunk.__new__(
            PartitionVectorChunk
        )
        ret._handle = move(handle)
        ret._br = br
        return ret

    @staticmethod
    def from_message(Message message not None, BufferResource br not None):
        """
        Construct a PartitionVectorChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PartitionVectorChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new PartitionVectorChunk extracted from the given message.
        """
        return PartitionVectorChunk.from_handle(
            make_unique[cpp_PartitionVectorChunk](
                message._handle.release[cpp_PartitionVectorChunk]()
            ),
            br,
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this PartitionVectorChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PartitionVectorChunk
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this PartitionVectorChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PartitionVectorChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_PartitionVectorChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PartitionVectorChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionVectorChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PartitionVectorChunk] release_handle(self):
        """
        Release ownership of the underlying C++ PartitionVectorChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionVectorChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
