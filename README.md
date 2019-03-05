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

    mpirun -np 4 python example_distributed_optimizer.py
