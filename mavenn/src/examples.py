"""examples.py: Functions to interface with built-in examples."""
# Standard imports
import pandas as pd
import numpy as np
import re
import glob
import os

# MAVE-NN imports
import mavenn
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import load


@handle_errors
def list_tutorials():
    """
    Reveal local directory where MAVE-NN tutorials are stored, as well as
    the names of available tutorial notebook files.
    """
    tutorials_dir = os.path.abspath(mavenn.__path__[0] + '/examples/tutorials')

    tutorial_file_names = glob.glob(f'{tutorials_dir}/*.ipynb')
    tutorial_base_names = [file_name.split('/')[-1]
                            for file_name in tutorial_file_names]
    tutorial_base_names.sort()

    print(f"The following MAVE-NN tutorials are available"
          f" (as Jupyter notebooks):\n")
    for name in tutorial_base_names:
        print(f'\t{name}')
    print(f"\nThese tutorial files are located in\n\n\t{tutorials_dir}/\n")


@handle_errors
def run_demo(name=None, print_code=False, print_names=True):
    """
    Perform demonstration of MAVE-NN.

    Parameters
    ----------
    name: (str, None)
        Name of demo to run. If ``None``, a list of valid demo names
        will be returned.

    print_code: (bool)
        If ``True``, the text of the demo file will be printed along with
        the output from running this file. If ``False``, only the demo output
        will be shown.

    print_names: (bool)
        If True and ``name=None``, the names of all demos will be printed.

    Returns
    -------
    demo_names: (list, None)
        List of demo names, returned if user passes ``names=None``.
        Otherwise None.
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
        if print_names:
            print("To run a demo, execute\n\n\t>>> mavenn.run_demo(name)"
                  "\n\nwhere 'name' is one of the following strings:\n")
            for i, name in enumerate(demo_names):
                print(f'\t{i+1}. "{name}"')
            print(f"\nPython code for each demo is located "
                  f"in\n\n\t{demos_dir}/\n")
        return demo_names

    # If input is valid, run run_demo
    elif name in demo_names:

        # Get run_demo file name
        file_name = demos_dict[name]

        # Print code if requested
        if print_code:
            with open(file_name, 'r') as f:
                content = f.read()
                line = '-------------------------------------------------------'
                print('Running %s:\n%s\n%s\n%s' % \
                      (file_name, line, content, line))
        else:
            print(f'Running {file_name}...')

        # Run run_demo
        exec(open(file_name).read())

        print('Done!')

    # If input is invalid, raise error
    else:
        # raise error
        check(False,
              f'name = {name} is invalid. Must be one of {demo_names}')


@handle_errors
def load_example_model(name=None):
    """
    Load an example model already inferred by MAVE-NN.

    Parameters
    -----------
    name: (str, None)
        Name of model to load. If ``None``, a list of valid model names
        will be printed.

    Returns
    -------
    model: (mavenn.Model)
        A pre-trained Model object.
    """
    models_dir = mavenn.__path__[0] +'/examples/models'
    model_file_names = glob.glob(f'{models_dir}/*.h5')
    models_dict = {}
    for file_name in model_file_names:
        base_name = file_name.split('/')[-1]
        pattern = '^(.*)\.h5$'
        key = re.match(pattern, base_name).group(1)
        models_dict[key] = file_name
    model_names = list(models_dict.keys())
    model_names.sort()

    # If no input, list valid names
    if name is None:
        print("Please enter a model name. Valid choices are:")
        print("\n".join([f'"{name}"' for name in model_names]))
        model = None

    # If input is valid, load model
    elif name in model_names:
        model = load(models_dir + '/' + name)

    # Otherwise
    else:
        check(False, f"Invalid choice of name={repr(name)}."
                     f"Please enter None or one of {model_names}")

    return model


@handle_errors
def load_example_dataset(name=None):
    """
    Load example dataset provided with MAVE-NN.

    Parameters
    ----------
    name: (str)
        Name of example dataset. If ``None``, a list of valid dataset names will
        be printed.

    Returns
    -------
    data_df: (pd.DataFrame)
        Dataframe containing the example datase.
    """
    # Create list of valid dataset names
    #dataset_names = ['mpsa', 'sortseq', 'gb1']
    #dataset_names.sort()

    # Set dataset_dir
    dataset_dir = mavenn.__path__[0] + '/examples/datasets/'

    dataset_file_names = glob.glob(f'{dataset_dir}*_data.csv.gz')
    dataset_dict = {}
    for file_name in dataset_file_names:
        base_name = file_name.split('/')[-1]
        pattern = '^(.*)_data\.csv\.gz$'
        key = re.match(pattern, base_name).group(1)
        dataset_dict[key] = file_name
    dataset_names = list(dataset_dict.keys())
    dataset_names.sort()

    # If no input, list valid names
    if name is None:
        print("Please enter a dataset name. Valid choices are:")
        print("\n".join([f'"{name}"' for name in dataset_names]))
        return None

    elif name in dataset_names:
        data_df = pd.read_csv(dataset_dir + f'{name}_data.csv.gz')
        return data_df

    # Otherwise
    else:
        check(False, f"Invalid choice of name={repr(name)}."
                     f"Please enter None or one of {dataset_names}")
