# mpi-tensorflow
Implement MPI as tensorflow operation


## setup

    cd mpi-tensorflow
    # build inplace
    python setup.py build_ext --inplace
    # install 
    python setup.py install
    # use system ld rather than anaconda's ld

## run

using classic MPI_allreduce
   
    mpirun -np 4 python example_distributed_allreduce_optimizer.py

compress gradient using HSQ in [the paper is under review](https://xinyandai.github.io/#Publications)

    mpirun -np 4 python example_distributed_hsq_optimizer.py


compress gradient using HSQ in [the paper is under review](https://xinyandai.github.io/#Publications)

    mpirun -np 4 python example_distributed_nn_optimizer.py


compress gradient using [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://papers.nips.cc/paper/6768-qsgd-communication-efficient-sgd-via-gradient-quantization-and-encoding)

    mpirun -np 4 python example_distributed_qsgd_optimizer.py
