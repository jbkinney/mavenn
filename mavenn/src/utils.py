from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from mavenn.src.error_handling import handle_errors, check
import matplotlib.pyplot as plt
import logomaker
import seaborn as sns
import pandas as pd
import mavenn

@handle_errors
def onehot_sequence(sequence, bases):

    """
    Encodes a single sequence into a one-hot matrix

    parameters
    ----------

    sequence: (str)
        string that needs to be turned into a one-hot encoded matrix

    bases: (list)
        specifies unique characters in the sequence


    returns
    -------
    oh_encoded_vector: (np.array)
        one-hot encoded array for the input sequence.


    """

    # sklearn objects and operations need for one-hot encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(bases)
    tmp = label_encoder.transform(bases)
    tmp = tmp.reshape(len(tmp), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(tmp)

    # perform one-hot encoding:
    categorical_vector = label_encoder.transform(list(sequence))

    # reshape so that array has correct dimensions for input into tf.
    categorical_vector = categorical_vector.reshape(len(categorical_vector), 1)
    oh_encoded_vector = onehot_encoder.transform(categorical_vector)
    return oh_encoded_vector


@handle_errors
def onehot_encode_array(data, bases_dict, ohe_single_batch_size=10000):

    """
    one-hot encode sequences in batches in a vectorized way

    parameters
    ----------

    data: (array-like)
        data which will be one-hot encoded

    bases_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein']

    ohe_single_batch_size: (int)
        integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large.

    returns
    -------
    input_seqs_ohe: (np array)
        array of one-hot encoded sequences based on the input data


    """

    # validate that sequences is array=like
    check(isinstance(data, (tuple, list, np.ndarray)),
          'type(data) = %s; data must be array-like.' %
          type(data))

    # check that ohe_single_batch_size is an integer
    check(isinstance(ohe_single_batch_size, (int, np.int64)),
          'type(ohe_single_batch_size) = %s must be of type int or numpy.int64' % type(ohe_single_batch_size))

    sequence_length = len(data[0])

    # container list for batches of oh-encoded sequences
    input_seqs_ohe_batches = []

    # partitions of batches
    ohe_batches = np.arange(0, len(data), ohe_single_batch_size)
    for ohe_batch_index in range(len(ohe_batches)):
        if ohe_batch_index == len(ohe_batches) - 1:
            # OHE remaining sequences (that are smaller than batch size)
            input_seqs_ohe_batches.append(
                onehot_sequence(''.join(data[ohe_batches[ohe_batch_index]:]), bases=bases_dict)
                    .reshape(-1, sequence_length, len(bases_dict)))
        else:
            # OHE sequences in batches
            input_seqs_ohe_batches.append(onehot_sequence(
                ''.join(data[ohe_batches[ohe_batch_index]:ohe_batches[ohe_batch_index + 1]]), bases=bases_dict)
                                          .reshape(-1, sequence_length, len(bases_dict)))

    # this array will contain the one-hot encoded sequences
    input_seqs_ohe = np.array([])

    # concatenate all the oh-encoded batches
    for batch_index in range(len(input_seqs_ohe_batches)):
        input_seqs_ohe = np.concatenate([input_seqs_ohe, input_seqs_ohe_batches[batch_index]
                                             .ravel()]).copy()

    # reshape so that shape of oh-encoded array is [number samples, sequence_length*alphabet_dict]
    input_seqs_ohe = input_seqs_ohe.reshape(len(data), sequence_length * len(bases_dict)).copy()

    return input_seqs_ohe


@handle_errors
def _generate_nbr_features_from_sequences(sequences,
                                          alphabet_dict='dna'):

    """
    Method that takes in sequences are generates sequences
    with neighbor features

    parameters
    ----------

    sequences: (array-like)
        array contains raw input sequences

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein']

    returns
    -------
    nbr_sequences: (array-like)
        Data Frame of sequences where each row contains a sequence example
        with neighbor features

    """

    # validate that sequences is array=like
    check(isinstance(sequences, (tuple, list, np.ndarray)),
          'type(sequences) = %s; sequences must be array-like.' %
          type(sequences))

    # check that alphabet_dict is valid
    check(alphabet_dict in {'dna', 'rna', 'protein'},
          'alphabet_dict = %s; must be "dna", "rna", or "protein"' %
          alphabet_dict)

    if alphabet_dict == 'dna':
        bases = ['A', 'C', 'G', 'T']
    elif alphabet_dict == 'rna':
        bases = ['A', 'C', 'G', 'U']
    elif alphabet_dict == 'protein':

        # this should be called amino-acids
        # need to figure out way to deal with
        # naming without changing a bunch of
        # unnecessary refactoring.
        bases = ['A', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R',
                 'S', 'T', 'V', 'W', 'Y']

    # form neighbor dinucleotide features that will
    # be used to one-hot encode sequnces
    nbr_dinucleotides = []

    for i in range(len(bases)):
        for j in range(len(bases)):
            nbr_dinucleotides.append(bases[i] + bases[j])

    # one-hot encode di-nucleotide training set
    dinuc_seqs_OHE = []
    for _ in range(len(sequences)):
        # take current raw training sequence
        raw_sequence = sequences[_]

        # split it into di-nucleotide pairs
        di_nucl_pairs = [raw_sequence[i:i + 2] for i in range(0, len(raw_sequence) - 1, 1)]

        # get indices of where pairs occur so that these indices could be used to one-hot encode.
        list_of_nbr_indices = [nbr_dinucleotides.index(dn) for dn in di_nucl_pairs]

        # do One-hot encoding. Every time a pair from list 'nbr_dinucleotides'
        # appears at a position, put 1 there, otherwise zeros.
        tmp_seq = np.array(list_of_nbr_indices)
        OHE_dinucl_seq = np.zeros((tmp_seq.size, len(nbr_dinucleotides)))
        OHE_dinucl_seq[np.arange(tmp_seq.size), tmp_seq] = 1

        dinuc_seqs_OHE.append(OHE_dinucl_seq.ravel())

    return np.array(dinuc_seqs_OHE)


def _generate_all_pair_features_from_sequences(sequences,
                                               alphabet_dict='dna'):

    """
    Method that takes in sequences are generates sequences
    with all pair features

    parameters
    ----------

    sequences: (array-like)
        array contains raw input sequences

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
     allowed: ['dna','rna','protein']


    returns
    -------

    all_pairs_sequences: (array-like)
        Data Frame of sequences where each row contains a sequence example
        with all-pair features

    """

    # validate that sequences is array=like
    check(isinstance(sequences, (tuple, list, np.ndarray)),
          'type(sequences) = %s; sequences must be array-like.' %
          type(sequences))

    # check that alphabet_dict is valid
    check(alphabet_dict in {'dna', 'rna', 'protein'},
          'alphabet_dict = %s; must be "dna", "rna", or "protein"' %
          alphabet_dict)

    if alphabet_dict == 'dna':
        bases = ['A', 'C', 'G', 'T']
    elif alphabet_dict == 'rna':
        bases = ['A', 'C', 'G', 'U']
    elif alphabet_dict == 'protein':

        # this should be called amino-acids
        # need to figure out way to deal with
        # naming without changing a bunch of
        # unnecessary refactoring.
        bases = ['A', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R',
                 'S', 'T', 'V', 'W', 'Y']

    # form neighbor dinucleotide features that will
    # be used to one-hot encode sequnces
    allpair_dinucleotides = []

    for i in range(len(bases)):
        for j in range(len(bases)):
            allpair_dinucleotides.append(bases[i] + bases[j])

    # one-hot encode di-nucleotide training set
    allpairs_seqs_OHE = []
    for _ in range(len(sequences)):

        # take current raw training sequence
        raw_sequence = sequences[_]

        # split it into all nucleotide pairs
        all_nucl_pairs = []

        for i in range(len(raw_sequence)):
            for j in range(i + 1, len(raw_sequence)):
                all_nucl_pairs.append(raw_sequence[i] + raw_sequence[j])

        # get indices of where pairs occur so that these indices could be used to one-hot encode.
        list_of_allpair_indices = [allpair_dinucleotides.index(dn) for dn in all_nucl_pairs]

        # do One-hot encoding. Every time a pair from list 'allpair_dinucleotides'
        # appears at a position, put 1 there, otherwise zeros.
        tmp_seq = np.array(list_of_allpair_indices)
        OHE_dinucl_seq = np.zeros((tmp_seq.size, len(allpair_dinucleotides)))
        OHE_dinucl_seq[np.arange(tmp_seq.size), tmp_seq] = 1

        allpairs_seqs_OHE.append(OHE_dinucl_seq.ravel())

    return np.array(allpairs_seqs_OHE)


@handle_errors
def ge_plots_for_mavenn_demo(loss_history,
                             predictions,
                             true_labels,
                             sequences,
                             GE_model):
    """

    Helper function that plots predictions vs true labels,
    losses and the GE nonlinearity for mavenn's demo functions

    parameters
    ----------
    loss_history: (tf loss history/array-like)
        arrays containing loss values

    sequences: (array-like)
        sequences which will be used to make predictions

    predictions: (array-like)
        model predictions for the sequences entered

    true_labels: (array-like)
        true labels for the model predictions

    GE_model: (GlobalEpistasisModel object)
        trained GE model from which the GE nonlinearity will be plotted


    returns
    -------
    None

    """

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    ax[0].plot(loss_history.history['loss'], color='blue')
    ax[0].plot(loss_history.history['val_loss'], color='orange')
    ax[0].set_title('Model loss', fontsize=12)
    ax[0].set_ylabel('loss', fontsize=12)
    ax[0].set_xlabel('epoch', fontsize=12)
    ax[0].legend(['train', 'validation'])

    ax[1].scatter(predictions, true_labels, s=1, alpha=0.5, color='black')
    ax[1].set_ylabel('observations')
    ax[1].set_xlabel('predictions')

    # get ge nonlinear function
    GE_nonlinearity = GE_model.ge_nonlinearity(sequences)

    ax[2].plot(GE_nonlinearity[1], GE_nonlinearity[0], color='black')
    ax[2].scatter(GE_nonlinearity[2], true_labels, color='gray', s=1, alpha=0.5)

    ax[2].set_ylabel('observations')
    ax[2].set_xlabel('latent trait ($\phi$)')

    plt.tight_layout()
    plt.show()


@handle_errors
def na_plots_for_mavenn_demo(loss_history,
                             NAR,
                             noise_model,
                             phi_range):
    """

    Helper function that plots inferred additive parameters,
    losses and the NA noise model for mavenn's demo functions

    parameters
    ----------
    loss_history: (tf loss history/array-like)
        arrays containing loss values

    NAR: (NoiseAgnosticModel object)
        trained NA model from which additive parameters
        will be extracted and plotted as a sequence logo

    noise_model: (array-like)
        the inferred noise model which will be plotted
        as a heatmap


    returns
    -------
    None
    """

    # plot results
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    ax[0].plot(loss_history.history['loss'], color='blue')
    ax[0].plot(loss_history.history['val_loss'], color='orange')
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'])

    # make logo to visualize additive parameters
    ax[1].set_ylabel('additive parameters')
    ax[1].set_xlabel('position')

    theta_df = pd.DataFrame(NAR.nn_model().layers[1].get_weights()[0].reshape(39, 4),columns=['A', 'C', 'G', 'T'])

    additive_logo = logomaker.Logo(theta_df/np.sqrt(np.sum(theta_df.values.ravel()**2)),
                                   center_values=True,
                                   ax=ax[1])

    # view the inferred noise model as a heatmap
    noise_model_heatmap = sns.heatmap(noise_model.T, cmap='Greens', ax=ax[2])
    ax[2].invert_yaxis()
    ax[2].set_xticks(([0,int(len(phi_range)/2), len(phi_range)-2]), minor=False)
    ax[2].set_xticklabels(([str(phi_range[0]), 0, str(phi_range[len(phi_range)-1])]), minor=False)
    ax[2].set_ylabel(' bin numbers')
    ax[2].set_xlabel('latent trait ($\phi$)')

    plt.tight_layout()
    plt.show()

@handle_errors
def load_olson_data_GB1():

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


@handle_errors
def get_example_dataset(name='MPSA'):
    """

    Parameters:
    -----------

    name: (str)
        Name of example dataset. Must be one of
        ('MPSA', 'Sort-Seq', 'GB1-DMS')

    Returns:
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

        return mpsa_df['sequence'].values, np.log10(mpsa_df['values'].values)

    elif name == 'Sort-Seq':

        sequences = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt',
                               dtype='str')
        bin_counts = np.loadtxt(mavenn.__path__[0] + '/examples/datafiles/sort_seq/full-wt/bin_counts.txt')

        return sequences, bin_counts

    elif name == 'GB1-DMS':

        gb1_df = load_olson_data_GB1()
        return gb1_df['sequence'].values, gb1_df['values'].values
