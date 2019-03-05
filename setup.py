from setuptools import setup, Extension, find_packages

import tensorflow as tf
import os


mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpic++ --showme:link").read().strip().split(' ')

os.system(
    "g++ "
    "-shared "
    "tf_mpi/mpi_ops/mpi_ops/*.cc "
    "-o tf_mpi/mpi_ops/mpi_ops.so "
    "{} {} "
    "-fPIC {} {} -O2".format(
        " ".join(mpi_compile_args),
        " ".join(mpi_link_args),
        " ".join(tf.sysconfig.get_compile_flags()),
        " ".join(tf.sysconfig.get_link_flags())
    )
)

my_libs_module = Extension(
    'tf_mpi.mpi_ops.mylibs',
    sources=['tf_mpi/mpi_ops/mylibsmodule.cc'],
    language="c++",
    extra_compile_args = mpi_compile_args,
    extra_link_args    = mpi_link_args,
)

setup(
    name='tf_mpi',
    version='0.1.1',
    keywords='tensorflow mpi',
    description='Implement MPI as tensorflow operation',
    license='MIT License',
    url='https://github.com/xinyandai/mpi-tensorflow',
    author='Xinyan DAI',
    author_email='xinyan.dai@@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=[],
    ext_modules=[my_libs_module]
)