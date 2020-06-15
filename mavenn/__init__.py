# Classes / functions imported with mavenn
from mavenn.src.model import Model
from mavenn.src.UI import GlobalEpistasisModel
from mavenn.src.UI import NoiseAgnosticModel
from mavenn.src.error_handling import handle_errors, check

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
        Must be one of {'GEmpsa, GEGB1, GEmpsaPairwise, NAsortseq'}.

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

