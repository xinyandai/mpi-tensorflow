import os
import tensorflow as tf
import numpy as np
from scipy import stats
from .myutils import normalize
from .myutils import fvecs_read
from .distributed_optimizer_compressor import IdenticalCompressor
from .distributed_optimizer_compressor import DistributedCompressorOptimizer
import tf_mpi.mpi_ops as mpi_wrappers

worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class NearestNeighborCompressor(object):
    def __init__(self, shape, c_dim, k):
        self.shape = shape
        self.dim = c_dim
        self.K = k
        if self.K == self.dim:
            self.codewords = stats.ortho_group.rvs(self.dim).astype(np.float32)
        else:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            fvecs = fvecs_read('{}/codebook/angular_dim_{}_Ks_{}.fvecs'.format(this_dir, self.dim, self.K))
            _, self.codewords = normalize(fvecs)

        self.c_dagger = np.linalg.pinv(self.codewords.T)
        self.code_dtype = tf.uint8 if self.K <= 2 ** 8 \
            else (tf.uint16 if self.K <= 2 ** 16 else tf.uint32)

    def compress(self, vec):
        """
        :param vec:
        :return: [u, codes]
            u : one dimension tensor of tf.float32
            codes: one dimension tensor of self.code_dtype
        """
        vec = tf.reshape(vec, (-1, self.dim))

        # calculate probability, complexity: O(d*K)
        # p = tf.transpose(tf.matmul(self.codewords, tf.transpose(vec)))
        p = tf.transpose(tf.matmul(self.c_dagger, tf.transpose(vec)))
        probability = tf.abs(p)

        # choose codeword with largest inner product
        codes = tf.cast(tf.argmax(probability, axis=1), tf.int32)

        u = tf.gather_nd(
            p, tf.stack([tf.range(p.shape[0]), codes], axis=1))
        return [u, tf.cast(codes, self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        codes = tf.cast(tf.reshape(codes, [-1]), tf.int32)
        norms = tf.reshape(norms, [-1])
        vec = tf.gather(self.codewords, codes)
        recover = tf.transpose(tf.multiply(tf.transpose(vec), norms))
        return recover

    def reduce_mean(self, grad, pre_operation):
        signs = self.compress(grad)
        received_signs = []
        for sign in signs:
            pre_operation = mpi_wrappers.mpi_ops_tf_allgather(
                sign, pre_operation)
            received_signs.append(pre_operation)

        recover = self.decompress(received_signs)
        averaged = tf.reduce_mean(
            tf.reshape(recover, (worker_size, -1)), axis=0)
        return tf.reshape(averaged, self.shape), pre_operation


class DistributedNNOptimizer(DistributedCompressorOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False,
                 device_dense='', device_sparse=''):
        super(DistributedNNOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def create_compressor(self, g):
        size = tf.reshape(g, [-1]).shape.as_list()[0]
        if size < 1024:
            return IdenticalCompressor()
        else:
            return NearestNeighborCompressor(shape=g.shape, c_dim=32, k=256)

