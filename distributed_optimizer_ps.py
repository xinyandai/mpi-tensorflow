import logging
import tensorflow as tf

import mpi_ops as mpi_wrappers

from distributed_optimizer import DistributedOptimizer


worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class DistributedPSOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        super(DistributedPSOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def synchronize_grads(self, gradients):
        averaged_gradients = []
        with tf.name_scope(self._name + "_GatherReduce"):
            pre_operation = None
            gathered_tensors = []

            def root_rank(_layer):
                return _layer % worker_size

            with tf.device(self._device_dense):
                for layer, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        pre_operation = mpi_wrappers.mpi_ops_tf_gather(
                            grad, pre_operation, root=root_rank(layer))
                        gathered_tensors.append(pre_operation)
                    else:
                        gathered_tensors.append(None)

                averaged = [None if t is None
                            else t if root_rank(layer) != worker_index
                            else tf.reduce_sum(t, 0) / worker_size
                            for layer, t in enumerate(gathered_tensors)]

                for layer, (gradient, var) in enumerate(gradients):
                    if gradient is not None:
                        pre_operation = mpi_wrappers.mpi_ops_tf_broadcast(
                            averaged[layer], pre_operation, root=root_rank(layer))
                        averaged_gradients.append((pre_operation, var))
                    else:
                        averaged_gradients.append((None, var))
        return averaged_gradients
