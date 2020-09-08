# Standard imports
import pandas as pd
import numpy as np
import re
import glob

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

        mpsa_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
        mpsa_df = mpsa_df.dropna()
        mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

        #return mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values)
        #return mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values)

        data_df = pd.DataFrame()
        data_df['y'] = np.log10(mpsa_df['values'].values)
        data_df['x'] = mpsa_df['sequence'].values
        return data_df

    elif name == 'sortseq':

        # sequences = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt',
        #                        dtype='str')
        # bin_counts = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/bin_counts.txt')
        #
        # return sequences, bin_counts

        data_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/full-wt-sort_seq.csv', index_col=[0])

        sequences = data_df['seq'].values
        bin_counts = data_df['bin'].values
        ct_n = data_df['ct'].values

        #return sequences, bin_counts, ct_n

        data_df = pd.DataFrame()
        data_df['ct'] = bin_counts
        data_df['y'] = ct_n
        data_df['x'] = sequences
        return data_df

    elif name == 'gb1':

        gb1_df = _load_olson_data_GB1()
        #return gb1_df['sequence'].values, gb1_df['values'].values

        data_df = pd.DataFrame()
        data_df['y'] = gb1_df['values'].values
        data_df['x'] = gb1_df['sequence'].values
        return data_df

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

    # GB1 WT sequences
    WT_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

    # WT sequence library and selection counts.
    WT_input_count = 1759616
    WT_selection_count = 3041819

    # load double mutant data
    oslon_mutant_positions_data = pd.read_csv(mavenn.__path__[0] +
                                              '/examples/datafiles/gb1/oslon_data_double_mutants_ambler.csv',
                                              na_values="nan")

    # lists that will contain sequences and their values
    sequences = []
    enrichment = []

    for loop_index in range(len(oslon_mutant_positions_data)):

        # skip row 259455 containing, it contains bad data
        if loop_index == 259455:
            continue

        # get positions of double mutants in sequence
        mut_1_index = int(oslon_mutant_positions_data['Mut1 Position'][loop_index]) - 2
        mut_2_index = int(oslon_mutant_positions_data['Mut2 Position'][loop_index]) - 2

        # get identity of mutations
        mut_1 = oslon_mutant_positions_data['Mut1 Mutation'][loop_index]
        mut_2 = oslon_mutant_positions_data['Mut2 Mutation'][loop_index]

        # form full mutant sequence.
        temp_dbl_mutant_seq = list(WT_seq)
        temp_dbl_mutant_seq[mut_1_index] = mut_1
        temp_dbl_mutant_seq[mut_2_index] = mut_2

        if loop_index % 100000 == 0:
            print('generating data: %d out of %d' % (loop_index,len(oslon_mutant_positions_data)))

        # calculate enrichment for double mutant sequence sequence
        input_count = oslon_mutant_positions_data['Input Count'][loop_index]
        selection_count = oslon_mutant_positions_data['Selection Count'][loop_index]
        # added pseudocount to ensure log doesn't throw up
        temp_fitness = ((selection_count + 1) / input_count) / (WT_selection_count / WT_input_count)

        # append sequence
        sequences.append(''.join(temp_dbl_mutant_seq))
        enrichment.append(temp_fitness)

    # load single mutants data
    oslon_single_mutant_positions_data = pd.read_csv(mavenn.__path__[0] +
                                                     '/examples/datafiles/gb1/oslon_data_single_mutants_ambler.csv',
                                                     na_values="nan")

    for loop_index in range(len(oslon_single_mutant_positions_data)):
        mut_index = int(oslon_single_mutant_positions_data['Position'][loop_index]) - 2

        mut = oslon_single_mutant_positions_data['Mutation'][loop_index]

        temp_seq = list(WT_seq)
        temp_seq[mut_index] = mut

        # calculate enrichment for sequence
        input_count = oslon_single_mutant_positions_data['Input Count'][loop_index]
        selection_count = oslon_single_mutant_positions_data['Selection Count'][loop_index]
        # added pseudo count to ensure log doesn't throw up
        temp_fitness = ((selection_count + 1) / input_count) / (WT_selection_count / WT_input_count)

        sequences.append(''.join(temp_seq))
        enrichment.append(temp_fitness)

    enrichment = np.array(enrichment).copy()

    gb1_df = pd.DataFrame({'sequence': sequences, 'values': np.log(enrichment)}, columns=['sequence', 'values'])
    return gb1_df


#TODO: Write this function. Subsumes load_example_dataset and load_example_model
@handle_errors
def load_example(name=None, model=True, training_data=False, test_data=False):
    """

    parameters
    ----------

    name: (str)
        Name of example model. If None, a list of valid
        choices will be printed.

    model: (bool)
        Whether to include inferred model.

    training_data: (bool)
        Whether to include training data. Slows execution.

    test_data: (bool)
        Whether to include test data. Slows execution.

    returns
    -------
    out_dict: (dict)
        A dictionary containing the example analysis.
        Keys may include,
            "model": The inferred MAVE-NN model
            "x_train": The training-set sequences
            "y_train": The training-set labels
            "x_test": The test-set sequeces
            "y_text": The test-set labels
            "wt_seq": The wild-type sequence, if applicable

    """
    out_dict = {}

    check(False, 'Function under construction.')

    return out_dict