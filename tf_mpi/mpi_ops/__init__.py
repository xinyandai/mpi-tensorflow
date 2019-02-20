import tensorflow as tf
import os
import sys

"""
import basic mpi functions
"""
mpilibs_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(mpilibs_dir)
import mylibs as mpilibs


def mpi_initialize():
    return mpilibs.mpi_initialize()


def mpi_finalize():
    return mpilibs.mpi_finalize()


def mpi_rank():
    return mpilibs.mpi_rank()


def mpi_size():
    return mpilibs.mpi_size()


worker_size = mpi_size()
worker_rank = mpi_rank()


"""
import mpi functions as tensorflow operations:
"""

mpilibs_dir = os.path.dirname(os.path.abspath(__file__))
mpi_ops = tf.load_op_library(os.path.join(mpilibs_dir, 'mpi_ops.so'))


def mpi_ops_tf_allgather(sendbuf, pre_node):
    pre_node = sendbuf if pre_node is None else pre_node
    return mpi_ops.tf_allgather(sendbuf, pre_node, size=worker_size)


def mpi_ops_tf_gather(sendbuf, pre_node, root):
    pre_node = sendbuf if pre_node is None else pre_node
    return mpi_ops.tf_gather(sendbuf, pre_node, root=root, rank=worker_rank, size=worker_size)


def mpi_ops_tf_broadcast(buff, pre_node, root):
    pre_node = buff if pre_node is None else pre_node
    return mpi_ops.tf_broadcast(buff, pre_node, root=root, size=worker_size)


def mpi_ops_tf_allreduce(sendbuf, pre_node):
    pre_node = sendbuf if pre_node is None else pre_node
    return mpi_ops.tf_allreduce(sendbuf, pre_node)
