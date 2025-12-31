from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "lssvr_optimized",
        ["lssvr_optimized.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["blas", "lapack"],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
