from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "legendre_matrices_cython",
        ["legendre_matrices.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)