# Classes / functions imported with mavenn
from mavenn.src.model import Model

from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import get_example_dataset
from mavenn.tests.functional_tests_mavenn import run_tests

from mavenn.src.utils import load
from mavenn.src.utils import estimate_instrinsic_information

from mavenn.src.utils import SkewedTNoiseModel
from mavenn.src.utils import GaussianNoiseModel
from mavenn.src.utils import CauchyNoiseModel

from mavenn.src import entropy_estimators as ee

from mavenn.src.utils import heatmap
from mavenn.src.utils import get_1pt_variants

# imports required for helper functions in demos
import pandas as pd
import mavenn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import re


@handle_errors
def demo(name='GEmpsa'):

    """
    Performs a demonstration of the mavenn software.

    parameters
    -----------

    name: (str)
        Must be one of {'GEmpsa, GEGB1, GEmpsaTrain, NAsortseq'}.

    returns
    -------
    None.

    """

    # build list of demo names and corresponding file names
    example_dir = '%s/examples' % os.path.dirname(__file__)
    all_base_file_names = os.listdir(example_dir)

    example_file_names = ['%s/%s' % (example_dir, temp_name)
                     for temp_name in all_base_file_names
                     if re.match('demo_.*\.py', temp_name)]

    examples_dict = {}
    for file_name in example_file_names:
        key = file_name.split('_')[-1][:-3]
        examples_dict[key] = file_name

    # check that name is valid
    check(name in examples_dict.keys(),
          'name = %s is not valid. Must be one of %s'
          % (repr(name), examples_dict.keys()))

    # open and run example file
    file_name = examples_dict[name]
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s' % \
              (file_name, line, content, line))
    exec(open(file_name).read())

