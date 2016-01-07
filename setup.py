import sys
import os
import re

from setuptools import setup
from setuptools import Extension



if (sys.version_info[0], sys.version_info[1]) != (2, 7):
    raise RuntimeError('sortseq is currently only compatible with Python 2.7.\nYou are using Python %d.%d' % (sys.version_info[0], sys.version_info[1]))


# main setup command
setup(
    name = 'sst', 
    description = 'Tools for analysis of Sort-Seq experiments.',
    version = '0.0.1',
    #long_description = readme,
    install_requires = [\
        'biopython>=1.6',\
        'scipy>=0.13',\
        'numpy>=1.7',\
        'pymc>=2.3.4',\
        'scikit-learn>=0.15.2',\
        'statsmodels>=0.5.0',\
        'mpmath>=0.19',\
        'pandas>=0.16.0',\
        'weblogo>=3.4'\
        ],
    platforms = 'Linux (and maybe also Mac OS X).',
    packages = ['sst'],
    package_dir = {'sst':'src'},
    download_url = 'https://github.com/jbkinney/sortseq/tarball/0.1',
    scripts = [
            'scripts/sst'
            ],
    #package_data = {'sst':['../data/sortseq/crp-wt/*.txt']}, # template from weblogo version 3.4
)

