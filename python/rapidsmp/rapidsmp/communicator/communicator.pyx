# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move


# Since a rapids::Communicator::Logger doesn't have a default ctor, we use
# these C++ functions to call the logger instance. This way Cython doesn't
# try to use the non-existent default ctor.
cdef extern from *:
    """
    template<typename T>
    void cpp_log_warn(std::shared_ptr<rapidsmp::Communicator> comm, T && msg) {
        comm->logger().warn(msg);
    }
    template<typename T>
    void cpp_log_info(std::shared_ptr<rapidsmp::Communicator> comm, T && msg) {
        comm->logger().info(msg);
    }
    template<typename T>
    void cpp_log_debug(std::shared_ptr<rapidsmp::Communicator> comm, T && msg) {
        comm->logger().debug(msg);
    }
    template<typename T>
    void cpp_log_trace(std::shared_ptr<rapidsmp::Communicator> comm, T && msg) {
        comm->logger().trace(msg);
    }
    int cpp_verbosity_level(std::shared_ptr<rapidsmp::Communicator> comm) {
        return comm->logger().verbosity_level();
    }
    """
    void cpp_log_warn[T](shared_ptr[cpp_Communicator] comm, T msg) except +
    void cpp_log_info[T](shared_ptr[cpp_Communicator] comm, T msg) except +
    void cpp_log_debug[T](shared_ptr[cpp_Communicator] comm, T msg) except +
    void cpp_log_trace[T](shared_ptr[cpp_Communicator] comm, T msg) except +
    uint32_t cpp_verbosity_level(shared_ptr[cpp_Communicator] comm) except +

cdef class Logger:
    """
    Logger.

    To control the verbosity level, set the environment variable `RAPIDSMP_LOG`:
      - `0`: Disable all logging.
      - `1`: Enable warnings only.
      - `2`: Enable warnings and informational messages (default).
      - `3`: Enable warnings, informational, and debug messages.
      - `4`: Enable warnings, informational, debug, and trace messages.
    """

    def __init__(self):
        raise TypeError("Please get a `Logger` from a communicater instance")

    @property
    def verbosity_level(self):
        """
        Get the verbosity level of the logger.

        Returns
        -------
            The verbosity level.
        """
        return cpp_verbosity_level(self._comm._handle)

    def warn(self, msg: str):
        """
        Logs a warning message.

        Formats and outputs a warning message if the verbosity level is `1` or higher.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        cpp_log_warn(self._comm._handle, move(_msg))

    def info(self, msg: str):
        """
        Logs an informational message.

        Formats and outputs an informational message if the verbosity level is `2`
        or higher.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        cpp_log_info(self._comm._handle, move(_msg))

    def debug(self, msg: str):
        """
        Logs a debug message.

        Formats and outputs a debug message if the verbosity level is `3` or higher.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        cpp_log_debug(self._comm._handle, move(_msg))

    def trace(self, msg: str):
        """
        Logs a trace message.

        Formats and outputs a trace message if the verbosity level is `4` or higher.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        cpp_log_trace(self._comm._handle, move(_msg))


cdef class Communicator:
    """
    Abstract base class for a communication mechanism between nodes.

    Provides an interface for sending and receiving messages between nodes,
    supporting asynchronous operations, GPU data transfers, and custom logging.
    Concrete implementations must define the virtual methods to enable specific
    communication backends.

    Notes
    -----
    This class is designed as an abstract base class, meaning it cannot be
    instantiated directly. Subclasses are required to implement the necessary
    methods to support the desired communication backend and functionality.
    """
    def __init__(self):
        raise TypeError(
            "Communicator is an abstract base case, please create a "
            "communicater through a concrete implementation such as "
            "`rapidsmp.mpi.new_communicator()`"
        )

    def __cinit__(self):
        self._logger = Logger.__new__(Logger)
        self._logger._comm = self

    @property
    def rank(self):
        """
        Get the rank of this communication node.

        Returns
        -------
            The rank.
        """
        return deref(self._handle).rank()

    @property
    def nranks(self):
        """
        Get the total number of ranks.

        Returns
        -------
            Total number of ranks.
        """
        return deref(self._handle).nranks()

    @property
    def logger(self):
        """
        Get the logger.

        Returns
        -------
            A logger instance.
        """
        return self._logger

    def get_str(self):
        """
        Get a string representation of the communicator.

        Returns
        -------
            A string describing the communicater
        """
        cdef string s = deref(self._handle).str()
        return s.decode('utf-8')
