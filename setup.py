import sys
import os
import re
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy
import glob
from setuptools import find_packages

if (sys.version_info[0], sys.version_info[1]) != (2, 7):
    raise RuntimeError('sortseq is currently only compatible with Python 2.7.\nYou are using Python %d.%d' % (sys.version_info[0], sys.version_info[1]))
input_data_list_commands = glob.glob('MPAthic_tests/commands/*.txt')
input_data_list_inputs = glob.glob('MPAthic_tests/input/*')

# DON'T FORGET THIS
ext_modules = Extension("MPAthic.fast",["src/fast.pyx"])

# main setup command
setup(
    name = 'MPAthic', 
    description = 'Tools for analysis of Sort-Seq experiments.',
    version = '0.01.01',
    author = 'Bill Ireland',
    author_email = 'wireland@caltech.edu',
    #long_description = readme,
    install_requires = [\
        'biopython>=1.6',\
        'pymc>=2.3.4',\
        'scikit-learn>=0.15.2, <= 0.16.1',\
        'statsmodels>=0.5.0',\
        'mpmath>=0.19',\
        'pandas>=0.16.0',\
        'weblogo>=3.4',\
        'Cython>=0.23.4',\
        'matplotlib<=1.5.0'
        ],
    platforms = 'Linux (and maybe also Mac OS X).',
    packages = ['MPAthic'] + find_packages(),
    package_dir = {'MPAthic':'src'},
    download_url = 'https://github.com/jbkinney/sortseq/tarball/0.1',
    scripts = [
            'scripts/MPAthic'
            ],
    zip_safe=False,
    ext_modules = cythonize(ext_modules),
    include_dirs=['.',numpy.get_include()],
    include_package_data=True,
    
    package_data = {
                     'MPAthic_tests.commands': ['*.txt'],
                     'MPAthic_tests': ['*.py','*.sh'],
                     'MPAthic_tests.input': ['*'],
                     'MPAthic_tests.output': ['*']
                 }
    #package_data = {'MPAthic':['tests/*']} # data for command line testing
)


