# mpi-tensorflow
Implement MPI as tensorflow operation


## setup

    cd mpi-tensorflow/mpi_ops
    python setup.py build_ext --inplace

## run

    mpirun -np 4 python example_distributed_optimizer.py
