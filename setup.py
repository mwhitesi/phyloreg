"""
---------------------------------------------------------------------
Copyright 2017 Alexandre Drouin and Faizy Ahsan
This file is part of phyloreg.
phyloreg is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
phyloreg is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with phyloreg.  If not, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------
"""
from platform import system as get_os_name

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []

# Required for the automatic installation of numpy
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

gradient_module = Extension('phyloreg._phyloreg',
                            sources=['cpp_extensions/phyloreg_python_bindings.cpp',
                                     'cpp_extensions/logistic.cpp'],
                            extra_compile_args=["-std=c++0x"] + os_compile_flags)

setup(
    name="phyloreg",
    version="0.1",
    author="Alexandre Drouin, Faizy Ahsan",
    author_email="",
    description='',
    license="GPL",
    keywords="",
    url="",

    packages=find_packages(),
    requires=['numpy', 'autograd'],

    cmdclass={'build_ext':build_ext},
    ext_modules=[gradient_module],

    test_suite='nose.collector',
    tests_require=['nose']
)
