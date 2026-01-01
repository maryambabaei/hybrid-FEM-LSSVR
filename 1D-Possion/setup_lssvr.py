from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

# Determine optimal compile flags based on platform
extra_compile_args = ['-O3', '-ffast-math']
if platform.machine() == 'x86_64':
    extra_compile_args.append('-march=native')

extensions = [
    Extension(
        "lssvr_kernels",
        ["lssvr_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=[],
    )
]

setup(
    name='lssvr_kernels',
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
    }),
    include_dirs=[np.get_include()],
)
