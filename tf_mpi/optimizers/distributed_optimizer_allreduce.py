import tensorflow as tf

import tf_mpi.mpi_ops as mpi_wrappers
from .distributed_optimizer import DistributedOptimizer


worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class DistributedAllReduceOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        super(DistributedAllReduceOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def synchronize_grads(self, gradients):
        averaged_gradients = []
        with tf.name_scope(self._name + "_Allreduce"):
            pre_operation = None
            with tf.device(self._device_dense):
                for layer, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        pre_operation = mpi_wrappers.mpi_ops_tf_allreduce(grad, pre_operation)
                        averaged_gradients.append((pre_operation / worker_size, var))
                    else:
                        averaged_gradients.append((None, var))

        return averaged_gradients

