# Standard imports
import pandas as pd
import numpy as np
import re
import glob

# MAVE-NN imports
import mavenn
from mavenn.src.error_handling import handle_errors, check


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



@handle_errors
def example_dataset(name='MPSA'):
    """

    Load example sequence-function datasets that
    come with the mavenn package.

    Parameters:
    -----------

    name: (str)
        Name of example dataset. Must be one of
        ('MPSA', 'Sort-Seq', 'GB1-DMS')

    Returns:
    --------
    X, y: (array-like)
        An array containing sequences X and an
        array containing their target values y
    """

    # check that parameter 'name' is valid
    check(name in {'MPSA', 'Sort-Seq', 'GB1-DMS'},
          'name = %s; must be "MPSA", "Sort-Seq", or "GB1-DMS"' %name)

    if name == 'MPSA':

        mpsa_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/mpsa/psi_9nt_mavenn.csv')
        mpsa_df = mpsa_df.dropna()
        mpsa_df = mpsa_df[mpsa_df['values'] > 0]  # No pseudocounts

        #return mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values)
        return mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values)

    elif name == 'Sort-Seq':

        # sequences = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt',
        #                        dtype='str')
        # bin_counts = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/bin_counts.txt')
        #
        # return sequences, bin_counts

        data_df = pd.read_csv(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/full-wt-sort_seq.csv', index_col=[0])

        sequences = data_df['seq'].values
        bin_counts = data_df['bin'].values
        ct_n = data_df['ct'].values

        return sequences, bin_counts, ct_n

    elif name == 'GB1-DMS':

        gb1_df = _load_olson_data_GB1()
        return gb1_df['sequence'].values, gb1_df['values'].values


@handle_errors
def _load_olson_data_GB1():

    """
    Helper function to turn data provided by Olson et al.
    into sequence-values arrays. This method is used in the
    GB1 GE demo.


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
