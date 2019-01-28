from distutils.core import setup, Extension
import os

os.environ["CC"] = "g++"

my_libs_module = Extension(
    'mylibs',
    language="c++",
    include_dirs="",
    sources=['mylibsmodule.cc'],
    extra_compile_args=[
        '-std=c++11',
        '-I/usr/local/include',
        '-Wl,-rpath',
        '-L/usr/local/lib',
        '-pthread',
        '-L/usr/lib/x86_64-linux-gnu/openmpi/lib'
        '-lmpi',
        '-pthread',
    ],
    library_dirs=['/usr/local/lib', '--enable-new-dtags'],
    libraries=['pthread', 'mpi']
)

setup(
    name='mylibs',
    version='1.0',
    description='This is my clibs package',
    ext_modules=[my_libs_module]
)
