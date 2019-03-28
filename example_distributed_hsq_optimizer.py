import logging

from example_model import Timer
from example_model import SimpleCNN
from example_model import download_mnist_retry

from tf_mpi.optimizers import DistributedHSQOptimizer

import tensorflow as tf
import tf_mpi.mpi_ops as mpi_wrappers


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    worker_index = mpi_wrappers.mpi_rank()
    worker_size = mpi_wrappers.mpi_size()


    i = 0
    batch_size = 64

    data = download_mnist_retry(seed=worker_index)
    net = SimpleCNN(learning_rate=1e-4 * worker_size, DistributedOptimizer=DistributedHSQOptimizer)


    timer = Timer()

    with tf.train.MonitoredTrainingSession(hooks=[]) as mon_sess:
        while True:
            # Compute and apply gradients.

            if i % 10 == 0:
                test_xs, test_ys = data.test.next_batch(1000)
                loss, accuracy = net.compute_loss_accuracy(test_xs, test_ys)
                logging.info("%d, %.3f, %.3f, %.3f" % (i, timer.toc(), loss, accuracy))
            i += 1
            xs, ys = data.train.next_batch(batch_size)
            net.compute_update(xs, ys)
