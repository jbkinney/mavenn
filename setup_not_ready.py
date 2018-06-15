import sys
import os
import re
from setuptools import setup
from setuptools import Extension
#from Cython.Build import cythonize
import numpy
import glob
from setuptools import find_packages
'''
'biopython',\
        'pymc>=2.3.4, < 3.0.0',\
        'scikit-learn>=0.15.2, <= 0.16.1',\
        'statsmodels>=0.5.0',\
        'mpmath>=0.19',\
        'pandas>=0.16.0',\
        'argparse',\
        'numpy',\
'''

if (sys.version_info[0], sys.version_info[1]) != (2, 7):
    raise RuntimeError('sortseq is currently only compatible with Python 2.7.\nYou are using Python %d.%d' % (sys.version_info[0], sys.version_info[1]))
input_data_list_commands = glob.glob('mpathic_tests/commands/*.txt')
input_data_list_inputs = glob.glob('mpathic_tests/input/*')

# DON'T FORGET THIS
ext_modules = Extension("src.fast",["src/fast.c"])

# main setup command
setup(
    name = 'src',
    description = 'Tools for analysis of Sort-Seq experiments.',
    version = '0.01.13',
    author = 'Bill Ireland',
    author_email = 'wireland@caltech.edu',
    #long_description = readme,
    install_requires = [\
        'scipy'
        ],
    platforms = 'Linux (and maybe also Mac OS X).',
    packages = ['src'] + find_packages(),
    package_dir = {'src':'src'},
    download_url = 'https://github.com/jbkinney/sortseq/tarball/0.1',
    scripts = [
            'scripts/mpathic'
            ],
    zip_safe=False,
    ext_modules = [ext_modules],
    include_dirs=['.',numpy.get_include()],
    include_package_data=True,
    
    package_data = {
                     'mpathic_tests.commands': ['*.txt'],
                     'mpathic_tests': ['*.py','*.sh'],
                     'mpathic_tests.input': ['*'],
                     'mpathic_tests.output': ['*']
                 }
    #package_data = {'mpathic':['tests/*']} # data for command line testing
)


