# Standard imports
import pandas as pd
import numpy as np
import re
import glob
from sklearn.model_selection import train_test_split

# MAVE-NN imports
import mavenn
from mavenn.src.error_handling import handle_errors, check


@handle_errors
def list_tutorials():
    """
    Reveals the local directory in which mavenn tutorials are stored.
    """
    tutorials_dir = mavenn.__path__[0] + '/examples/tutorials'
    tutorial_file_names = glob.glob(f'{tutorials_dir}/*.ipynb')
    tutorial_base_names = [file_name.split('/')[-1]
                            for file_name in tutorial_file_names]
    tutorial_base_names.sort()

    print(f"The following MAVE-NN tutorials are available (as Jupyter notebooks):\n")
    for name in tutorial_base_names:
        print(f'\t{name}')
    print(f"\nThese tutorial files are located in\n\n\t{tutorials_dir}/\n")


@handle_errors
def list_demos():
    """
    Reveals the local directory in which mavenn demos are stored.
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

    print("To run a demo, execute\n\n\t>>> mavenn.run_demo(name)"
          "\n\nwhere 'name' is one of the following strings:\n")
    for i, name in enumerate(demo_names):
        print(f'\t{i+1}. "{name}"')
    print(f"\nPython code for each demo is located in\n\n\t{demos_dir}/\n")


@handle_errors
def run_demo(name=None, print_code=False):
    """
    Performs a demonstration of the mavenn software.

    parameters
    -----------
    name: (str, None)
        Name of run_demo to run. If None, a list of valid run_demo names
        will be printed.

    print_code: (bool)
        If True, text of the run_demo file will be printed along with
        the output of the run_demo file. If False, only the run_demo output
        will be shown.

    return
    ------
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
        print("Please enter a run_demo name. Valid are:")
        print("\n".join([f'"{name}"' for name in demo_names]))

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

    parameters
    -----------
    name: (str, None)
        Name of model to load. If None, a list of valid model names
        will be printed.

    returns
    -------
    model: (mavenn.Model)

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
        model = mavenn.load(models_dir + '/' + name)

    # Otherwise
    else:
        check(False, f"Invalid choice of name={repr(name)}."
                     f"Please enter None or one of {model_names}")

    return model


@handle_errors
def load_example_dataset(name=None):
    """
    Load example dataset.

    Parameters:
    -----------

    name: (str)
        Name of example dataset. If None, a list
        of valid names will be printed.

    Returns:
    --------
    data_df: (np.array)
        Pandas dataframe containing the example data.
    """

    # Create list of valid dataset names
    dataset_names = ['mpsa', 'sortseq', 'gb1']
    dataset_names.sort()

    # If no input, list valid names
    if name is None:
        print("Please enter a dataset name. Valid choices are:")
        print("\n".join([f'"{name}"' for name in dataset_names]))
        return None

    elif name == 'mpsa':

        mpsa_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/mpsa/brca2_lib1_rep1.csv')

        data_df = pd.DataFrame()
        data_df['y'] = mpsa_df['log_psi'].values
        data_df['x'] = mpsa_df['ss'].values

        return data_df

    elif name == 'sortseq':

        # sequences = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt',
        #                        dtype='str')
        # bin_counts = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/bin_counts.txt')
        #
        # return sequences, bin_counts

        data_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/full-wt-sort_seq.csv', index_col=[0])

        sequences = data_df['seq'].values
        bins = data_df['bin'].values
        ct = data_df['ct'].values

        #return sequences, bin_counts, ct_n

        data_df = pd.DataFrame()
        data_df['y'] = bins
        data_df['ct'] = ct
        data_df['x'] = sequences
        return data_df

    elif name == 'gb1':

        gb1_df = _load_olson_data_GB1()
        #return gb1_df['sequence'].values, gb1_df['values'].values

        # data_df = pd.DataFrame()
        # data_df['y'] = gb1_df['values'].values
        # data_df['x'] = gb1_df['sequence'].values
        return gb1_df

    # Otherwise
    else:
        check(False, f"Invalid choice of name={repr(name)}."
                     f"Please enter None or one of {dataset_names}")


@handle_errors
def _load_olson_data_GB1():

    """
    Helper function to turn data provided by Olson et al.
    into sequence-values arrays. This method is used in the
    GB1 GE run_demo.


    return
    ------
    gb1_df: (pd dataframe)
        dataframe containing sequences (single and double mutants)
        and their corresponding log enrichment values

    """

    gb1_df = pd.read_csv(mavenn.__path__[0]+'/examples/datafiles/gb1/gb1_data.csv.gz')

    return gb1_df

@handle_errors
def load_example(which=None,
                 name=None):
    """

    Method that returns either a model or an example dataset
    based on the parameter "which".

    parameters
    ----------
    which: (str)
        String which specifies whether to load example model or
        example dataset. Valid choice include ['model', 'dataset'].

    name: (str)
        Name of example model or example dataset. If None, a list
        of valid choices will be printed.

    returns
    -------
    if which == "model":
        mavenn-model
    else:
        return data_df with either training or test data.

    """

    valid_which_list = ['model', 'training_data', 'test_data']

    # if which is none, list valid choices of model/datasets names that can be loaded.
    if which is None:

        print(f"Valid choices for parameter which must be one of {valid_which_list}")
        return None

    elif which == 'model':

        if name is None:
            # call example model which would list out valid model names
            load_example_model()
            return None
        else:
            return load_example_model(name=name)

    elif which == 'training_data':

        if name is None:
            # call example dataset which would list out valid dataset names
            load_example_dataset()
            return None

        # TODO: change this snippet when the dataset formats for all experiments are the same.
        elif name == 'gb1':

            data_df = load_example_dataset(name=name)

            gb1_df_training = data_df[data_df['training_set'] == True].copy()

            return gb1_df_training

        else:
            data_df = load_example_dataset(name=name)

            # Extract x and y as Numpy arrays
            x = data_df['x'].values
            y = data_df['y'].values

            # Split data 80/20 into training / test sets.

            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

            return pd.DataFrame({"x": x_train, "y": y_train})

    elif which == 'test_data':

        if name is None:
            # call example dataset which would list out valid dataset names
            load_example_dataset()
            return None

        # TODO: change this snippet when the dataset formats for all experiments are the same.
        elif name == 'gb1':

            data_df = load_example_dataset(name=name)
            gb1_df_test = data_df[data_df['training_set'] == False].copy()

            return gb1_df_test

        else:
            data_df = load_example_dataset(name=name)

            # Extract x and y as Numpy arrays
            x = data_df['x'].values
            y = data_df['y'].values

            # Split data into training / test sets.

            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

            return pd.DataFrame({"x": x_test, "y": y_test})

    else:
        check(which in valid_which_list, f"parameter which = {which}, "
                                         f"needs to be one of {valid_which_list}.")
