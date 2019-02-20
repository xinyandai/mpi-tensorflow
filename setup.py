from setuptools import setup, Extension, find_packages

import tensorflow as tf
import os

os.environ["CC"] = "g++"

os.system(
    "g++ -std=c++11 -shared tf_mpi/mpi_ops/mpi_ops/*.cc -o tf_mpi/mpi_ops/mpi_ops.so "
    "-pthread -lmpi -fPIC {} {} -O2".format(
        " ".join(tf.sysconfig.get_compile_flags()),
        " ".join(tf.sysconfig.get_link_flags())
    )
)

my_libs_module = Extension(
    'tf_mpi.mpi_ops.mylibs',
    language="c++",
    include_dirs="",
    sources=['tf_mpi/mpi_ops/mylibsmodule.cc'],
    extra_compile_args=[
        '-std=c++11',
        '-lmpi',
        '-pthread',
    ],
    library_dirs=['/usr/local/lib', '--enable-new-dtags'],
    libraries=['pthread', 'mpi']
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