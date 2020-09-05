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

from mavenn.src.utils import additive_heatmap
from mavenn.src.utils import pairwise_heatmap
from mavenn.src.utils import get_1pt_variants

# imports required for helper functions in demos
import pandas as pd
import mavenn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import re
import glob

@handle_errors
def demo(name=None, print_code=True):

    """
    Performs a demonstration of the mavenn software.

    parameters
    -----------
    name: (str, None)
        Name of demo to run. If None, a list of valid demo names
        will be printed.

    print_code: (bool)
        If True, text of the demo file will be printed along with
        the output of the demo file. If False, only the demo output
        will be shown.

    returns
    -------
    None

    """

    demos_dir = mavenn.__path__[0] +'/examples/demos'
    demo_file_names = glob.glob(f'{demos_dir}/*.py')
    demos_dict = {}
    for file_name in demo_file_names:
        base_name = file_name.split('/')[-1]
        pattern = '^(.*)\.py$'
        key = re.match(pattern, base_name).group(1)
        demos_dict[key] = file_name
    demo_names = list(demos_dict.keys())
    demo_names.sort()

    # If no input, list valid names
    if name is None:
        print("Please enter a demo name. Valid are:")
        print("\n".join([f'"{name}"' for name in demo_names]))

    # If input is valid, run demo
    elif name in demo_names:

        # Get demo file name
        file_name = demos_dict[name]

        # Print code if requested
        if print_code:
            with open(file_name, 'r') as f:
                content = f.read()
                line = '-------------------------------------------------------'
                print('Running %s:\n%s\n%s\n%s' % \
                      (file_name, line, content, line))
        else:
            pass

        # Run demo
        exec(open(file_name).read())

    # If input is invalid, raise error
    else:
        # raise error
        check(False,
              f'name = {name} is invalid. Must be one of {demo_names}')

