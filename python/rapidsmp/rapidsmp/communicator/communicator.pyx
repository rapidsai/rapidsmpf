# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move


# Since a rapids::Communicator::Logger doesn't have a default ctor, we use
# this C++ function to call the logger instance without having Cython try
# to use the non-existent default ctor.
cdef extern from *:
    """
    template<typename T>
    void cpp_log_warn(std::shared_ptr<rapidsmp::Communicator> comm, T && msg) {
        comm->logger().warn(msg);
    }
    """
    void cpp_log_warn[T](shared_ptr[cpp_Communicator] comm, T msg) except +


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

    def warn(self, msg: str):
        """
        Logs a warning message.

        Formats and outputs a warning message if the verbosity level is `1` or higher.

        Parameters
        ----------
        msg
            The warning message to log.
        """
        cdef string _msg = msg.encode()
        cpp_log_warn(self._comm._handle, move(_msg))


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
