import tensorflow as tf

import mpi_ops as mpi_wrappers
from distributed_optimizer_compressor import IdenticalCompressor
from distributed_optimizer_compressor import DistributedCompressorOptimizer

worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class QSGDCompressor(object):
    def __init__(self, shape, c_dim, s=256):
        self.s = s
        self.shape = shape
        self.dim = c_dim
        self.code_dtype = tf.int8 if self.s <= 2 ** 7 \
            else (tf.int16 if self.s <= 2 ** 15 else tf.int32)

    def compress(self, vec):
        vec = tf.reshape(vec, (-1, self.dim))
        norms = tf.linalg.norm(vec, axis=1)
        normalized_vecs = tf.div_no_nan(vec, norms[:, tf.newaxis])
        scaled_vec = tf.abs(normalized_vecs) * self.s
        l = tf.clip_by_value(scaled_vec, 0, self.s-1)
        l = tf.cast(l, tf.int32)

        # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
        probabilities = scaled_vec - tf.cast(l, tf.float32)
        comp = probabilities > tf.random.uniform(shape=l.shape, minval=0, maxval=1)
        l = l + tf.cast(comp, tf.int32)
        l = tf.multiply(tf.cast(tf.sign(vec), tf.int32), l)

        return [norms, tf.cast(l, self.code_dtype)]

    def decompress(self, signature):
        [norm, l] = signature

        norm = tf.reshape(norm, [-1])
        l = tf.reshape(l, [-1, self.dim])

        scaled_vec = tf.cast(l, tf.float32)
        compressed = tf.transpose(
            tf.multiply(tf.transpose(scaled_vec), norm) / self.s)
        return compressed

    def reduce_mean(self, grad, pre_operation):
        signs = self.compress(grad)
        received_signs = []
        for sign in signs:
            pre_operation = mpi_wrappers.mpi_ops_tf_allgather(
                sign, pre_operation)
            received_signs.append(pre_operation)

        recover = self.decompress(received_signs)
        averaged = tf.reduce_mean(tf.reshape(recover, (worker_size, -1)), axis=0)
        return tf.reshape(averaged, self.shape), pre_operation


class DistributedQSGDOptimizer(DistributedCompressorOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False,
                 device_dense='', device_sparse=''):
        super(DistributedQSGDOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def create_compressor(self, g):
        size = tf.reshape(g, [-1]).shape.as_list()[0]
        if size < 1024:
            return IdenticalCompressor()
        else:
            return QSGDCompressor(g.shape, c_dim=256)
