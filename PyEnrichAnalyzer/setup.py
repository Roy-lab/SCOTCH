from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import glob
import pybind11

__version__ = "0.0.1"

# Your C++ source files here
source_files = glob.glob('*.C')  # adding all .C files to the source file list

ext_modules = [
    Extension(
        'pyEnrichAnalyzer',  # name of output module
        source_files,
        include_dirs=[  # locations of your includes here
            '/usr/local/include',
            '/usr/local/include/python3.11m',
            '/usr/local/include/gsl/',
            pybind11.get_include(),
        ],
        libraries=['gsl'],  # add needed C/C++ libraries here
        library_dirs=['/usr/local', '/usr/local/lib'],
        language='c++',
        extra_compile_args=['-std=c++14']
    ),
]


class BuildExt(build_ext):
    def build_extensions(self):
        super(BuildExt, self).build_extensions()


setup(
    name='pyEnrichAnalyzer',
    version=__version__,
    author='',
    author_email='',
    url='',
    description='',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)