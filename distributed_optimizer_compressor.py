import tensorflow as tf
import numpy as np
from scipy import stats
from distributed_optimizer import DistributedOptimizer

import mpi_ops as mpi_wrappers

worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class IdenticalCompressor(object):
    @staticmethod
    def reduce_mean(grad, pre_operation):
        pre_operation = mpi_wrappers.mpi_ops_tf_allreduce(grad, pre_operation)
        return pre_operation / worker_size, pre_operation


class DistributedCompressorOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False,
                 device_dense='', device_sparse=''):
        super(DistributedCompressorOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def create_compressor(self, g):
        raise NotImplemented()

    def synchronize_grads(self, gradients):
        averaged_gradients = []
        compressors = [self.create_compressor(g) for g, _ in gradients]
        pre_operation = None
        for c, (grad, var) in zip(compressors, gradients):
            if grad is not None:
                averaged, pre_operation = c.reduce_mean(grad, pre_operation)
                averaged_gradients.append((averaged, var))
            else:
                averaged_gradients.append((None, var))
        return averaged_gradients
