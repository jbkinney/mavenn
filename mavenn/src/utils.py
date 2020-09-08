# Standard imports
import numpy as np
import pandas as pd
import mavenn

# scikit-learn imports (for one-hot encoding)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# tensorflow imports
import tensorflow.keras.backend as K
import tensorflow.keras

# Special function imports
from numpy import log as Log
from scipy.special import betaincinv as BetaIncInv
from scipy.special import gammaln as LogGamma
from scipy.stats import cauchy
from scipy.special import erfinv

# Imports from MAVE-NN
from mavenn.src.error_handling import handle_errors, check


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

    # check that data[0] passed has length, which will be used to get sequence length
    check(hasattr(data[0], '__len__'), 'entered sequence data does not have length, needs to have length > 0')

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

        #print(all_nucl_pairs)
        # do One-hot encoding. Every time a pair from list 'allpair_dinucleotides'
        # appears at a position, put 1 there, otherwise zeros.
        tmp_seq = np.array(list_of_allpair_indices)
        OHE_dinucl_seq = np.zeros((tmp_seq.size, len(allpair_dinucleotides)))
        OHE_dinucl_seq[np.arange(tmp_seq.size), tmp_seq] = 1

        allpairs_seqs_OHE.append(OHE_dinucl_seq.ravel())

    return np.array(allpairs_seqs_OHE)


@handle_errors
def _center_matrix(df):
    """
    Centers each row of a matrix about zero by subtracting out the mean.
    This method is used for gauge-fixing additive latent models

    parameters
    ----------

    df: (pd.DataFrame)
        a (L x C) pandas dataframe which contains additive latent model
        parameters

    returns
    -------
    out_df: (pd.DataFrame)
        dataframe that is mean centered.

    theta_bar: (array-like)
        mean of parameters at position l

    """

    # Compute mean value of each row
    means = df.mean(axis=1).values
    theta_bar = -df.mean(axis=1).values[:, np.newaxis]

    # Subtract out means
    out_df = df.copy()
    out_df.loc[:, :] = df.values - means[:, np.newaxis]

    return out_df, theta_bar


@handle_errors
def fix_gauge_additive_model(sequenceLength,
                             alphabetSize,
                             theta):
    """
    Calculates the hierarchical gauge where constant feature > single-site features.
    Model parameters are given in the 1d array theta. Order of coefficients in theta, e.g. ,
    sequence length = 3 and alphabet = 'AB': theta0,theta1A,theta1B,theta2A,...

    parameters
    ----------
    sequenceLength: (int)
        length of one input sequence

    alphabetSize: (int)
        number of unique bases or amino-acids; e.g. 4 for DNA and 20 for proteins

    theta: (array-like)
        the parameters of the latent neigbhor model

    returns
    -------
    thetaGaugeFixed: (array-like)
        the gauge fixed neighbor parameters

    """

    # copy/reshape input parameter vector:
    thetaConstant = theta[0]
    thetaSinglesite = np.copy(theta[1:]).reshape(sequenceLength, alphabetSize)

    # mean center columns in position weight matrix
    thetaConstant += sum(thetaSinglesite.mean(axis=1))
    thetaSinglesite = thetaSinglesite - thetaSinglesite.mean(axis=1, keepdims=True)

    thetaGaugeFixed = np.concatenate([[thetaConstant], thetaSinglesite.ravel()])

    return thetaGaugeFixed


@handle_errors
def fix_gauge_neighbor_model(sequenceLength,
                             alphabetSize,
                             theta):
    """
    Calculates the hierarchical gauge where constant feature > single-site features > neighbor features.
    Model parameters are given in the 1d array theta. Order of coefficients in theta, e.g. sequence length = 3
    and alphabet = 'AB':  theta0,theta1A,theta1B,...,theta12AA,theta12AB,theta12BA,theta12BB,
    theta23AA,theta23AB,theta23BA,theta23BB

    parameters
    ----------
    sequenceLength: (int)
        length of one input sequence

    alphabetSize: (int)
        number of unique bases or amino-acids; e.g. 4 for DNA and 20 for proteins

    theta: (array-like)
        the parameters of the latent neigbhor model

    returns
    -------
    thetaGaugeFixed: (array-like)
        the gauge fixed neighbor parameters

    """

    # copy/reshape input parameter vector:
    numSinglesiteFeatures = sequenceLength * alphabetSize
    numPosPairs = sequenceLength - 1
    thetaConstant = theta[0]
    thetaSinglesite = np.copy(theta[1:numSinglesiteFeatures + 1]).reshape(sequenceLength, alphabetSize)
    thetaPairwise = theta[numSinglesiteFeatures + 1:].reshape(numPosPairs, alphabetSize, alphabetSize)

    # apply g1 (given in SI):
    thetaConstant += sum(thetaPairwise.mean(axis=(1, 2)))
    thetaPairwise = thetaPairwise - thetaPairwise.mean(axis=(1, 2), keepdims=True)

    # apply g2 and g3 (given in SI) to pairwise coefficients:
    rowMeans = np.mean(thetaPairwise, axis=2, keepdims=True)
    columnMeans = np.mean(thetaPairwise, axis=1, keepdims=True)
    thetaPairwise = thetaPairwise - rowMeans - columnMeans

    # apply g2 and g3 (given in SI) to single-site coefficients:
    for p in range(1, sequenceLength):
        thetaSinglesite[p] = thetaSinglesite[p] + columnMeans[p - 1].ravel()
    for p in range(sequenceLength - 1):
        thetaSinglesite[p] = thetaSinglesite[p] + rowMeans[p].ravel()

    # apply g4 (given in SI):
    thetaConstant += sum(thetaSinglesite.mean(axis=1))
    thetaSinglesite = thetaSinglesite - thetaSinglesite.mean(axis=1, keepdims=True)

    thetaGaugeFixed = np.concatenate([[thetaConstant], thetaSinglesite.ravel(), thetaPairwise.ravel()])

    return thetaGaugeFixed


@handle_errors
def kronecker_delta(n1, n2):
    """
    helper method implementing Kronecker delta,
    used in fix_gauge_pairwise_model
    """

    if n1 == n2:
        return 1
    else:
        return 0


@handle_errors
def fix_gauge_pairwise_model(sequenceLength,
                             alphabetSize,
                             theta):
    """
    Calculates the hierarchical gauge where constant feature > single-site features > pairwise features.

    parameters
    ----------
    sequenceLength: (int)
        length of one input sequence

    alphabetSize: (int)
        number of unique bases or amino-acids; e.g. 4 for DNA and 20 for proteins

    theta: (array-like)
        the parameters of the latent neigbhor model

    returns
    -------
    thetaGaugeFixed: (array-like)
        the gauge fixed pairwise parameters

     """

    """"""

    # copy/reshape input parameter vector:
    numSinglesiteFeatures = sequenceLength * alphabetSize
    numPosPairs = int(sequenceLength * (sequenceLength - 1) / 2)
    thetaConstant = theta[0]
    thetaSinglesite = np.copy(theta[1:numSinglesiteFeatures + 1]).reshape(sequenceLength, alphabetSize)
    thetaPairwise = theta[numSinglesiteFeatures + 1:].reshape(numPosPairs, alphabetSize, alphabetSize)

    # apply g1 (given in SI):
    thetaConstant += sum(thetaPairwise.mean(axis=(1, 2)))
    thetaPairwise = thetaPairwise - thetaPairwise.mean(axis=(1, 2), keepdims=True)

    # apply g2 and g3 (given in SI) to pairwise coefficients:
    rowMeans = np.mean(thetaPairwise, axis=2, keepdims=True)
    columnMeans = np.mean(thetaPairwise, axis=1, keepdims=True)
    thetaPairwise = thetaPairwise - rowMeans - columnMeans

    # apply g2 and g3 (given in SI) to single-site coefficients:
    l = [i for i in range(1, sequenceLength)]
    splittingPoints = [sum(l[-k:]) for k in range(1,
                                                  sequenceLength)]  # [(sequenceLength-1), (sequenceLength-1+sequenceLength-2),...,(sequenceLength-1+sequenceLength-2+...+1)]
    rowMeans = np.array(np.split(rowMeans, splittingPoints, axis=0),
                        dtype=object)  # reshape rowMeans so that it is indexed as [pos1][pos2-(pos1+1),char1,char2]
    columnMeans = np.array(np.split(columnMeans, splittingPoints, axis=0),
                           dtype=object)  # reshape columnMeans so that it is indexed as [pos1][pos2-(pos1+1),char1,char2]
    for p in range(sequenceLength):
        for pp in range(p):
            thetaSinglesite[p] = thetaSinglesite[p] + columnMeans[pp][p - (pp + 1)].ravel()
        for pp in range(p + 1, sequenceLength):
            thetaSinglesite[p] = thetaSinglesite[p] + rowMeans[p][pp - (p + 1)].ravel()

    # apply g4 (given in SI):
    thetaConstant += sum(thetaSinglesite.mean(axis=1))
    thetaSinglesite = thetaSinglesite - thetaSinglesite.mean(axis=1, keepdims=True)

    thetaGaugeFixed = np.concatenate([[thetaConstant], thetaSinglesite.ravel(), thetaPairwise.ravel()])

    return thetaGaugeFixed


class fixDiffeomorphicMode(tensorflow.keras.layers.Layer):

    """
    This layer fixes the diffeomorphic mode in the neural network
    model after the model parameters theta have been gauge-fixed.
    """

    def __init__(self, **kwargs):
        super(fixDiffeomorphicMode, self).__init__(**kwargs)

    def call(self, inputs):
        # inpus here are phi-hoc
        phi_mean = K.mean(inputs)
        phi_std = K.std(inputs)
        return (inputs - phi_mean) / phi_std
        return inputs


class SkewedTNoiseModel:

    """
    Class used to compute quantiles for the Skewed T noise model for
    GE regression. Usage example:

    quantiles = ComputeSkewedTQuantiles(GER,yhat_GE)

    The attributes gives +/- 1 sigma quantiles

    quantiles.plus_sigma_quantile
    quantiles.minus_sigma_quantile

    parameters
    ----------

    model: (mavenn.Model object)

         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Skewed T noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the quantiles will be computed.
        This should be the output of the GE model.


    user_quantile: (float between [0,1])
        If not None, the attribute user_quantile_values will contain quantile values
        for the user_quantile value specified
    """

    def __init__(self,
                 model,
                 yhat_GE,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat_GE = yhat_GE
        self.q = q

        polynomial_weights_a = self.model.get_nn().layers[8].get_weights()[0].copy()
        polynomial_weights_b = self.model.get_nn().layers[8].get_weights()[1].copy()
        polynomial_weights_s = self.model.get_nn().layers[8].get_weights()[2].copy()

        log_a = 0
        log_b = 0
        log_scale = 0

        for polynomial_index in range(len(polynomial_weights_a)):
            log_a += polynomial_weights_a[polynomial_index][0] * np.power(yhat_GE, polynomial_index)
            log_b += polynomial_weights_b[polynomial_index][0] * np.power(yhat_GE, polynomial_index)
            log_scale += polynomial_weights_s[polynomial_index][0] * np.power(yhat_GE, polynomial_index)

        self.plus_sigma_quantile = self.y_quantile(0.16, self.yhat_GE, np.exp(log_scale), np.exp(log_a), np.exp(log_b))
        self.minus_sigma_quantile = self.y_quantile(0.84, self.yhat_GE, np.exp(log_scale), np.exp(log_a), np.exp(log_b))

        self.log_a = log_a
        self.log_b = log_b
        self.log_scale = log_scale

        if q is not None:

            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(self.y_quantile(current_q,
                                                                 self.yhat_GE,
                                                                 np.exp(self.log_scale),
                                                                 np.exp(self.log_a),
                                                                 np.exp(self.log_b)).ravel())

            # self.user_quantile_values = self.y_quantile(self.q,
            #                                             self.yhat_GE,
            #                                             np.exp(log_scale),
            #                                             np.exp(log_a),
            #                                             np.exp(log_b))


    # First compute log PDF to avoid overflow problems
    def log_f(self, t, a, b):
        arg = t / np.sqrt(a + b + t ** 2)
        return (1 - a - b) * Log(2) + \
               -0.5 * Log(a + b) + \
               LogGamma(a + b) + \
               -LogGamma(a) + \
               -LogGamma(b) + \
               (a + 0.5) * Log(1 + arg) + \
               (b + 0.5) * Log(1 - arg)


    # Exponentiate to get true distribution
    def f(self, t, a, b):
        return np.exp(self.log_f(t, a, b))


    # Compute the mode as a function of a and b
    def t_mode(self, a, b):
        return (a - b) * np.sqrt(a + b) / (np.sqrt(2 * a + 1) * np.sqrt(2 * b + 1))


    # Compute mean
    def t_mean(self, a, b):
        if a <= 0.5 or b <= 0.5:
            return np.nan
        else:
            return 0.5 * (a - b) * np.sqrt(a + b) * np.exp(
                LogGamma(a - 0.5) + LogGamma(b - 0.5) - LogGamma(a) - LogGamma(b))


    # Compute variance
    def t_std(self, a, b):
        if a <= 1 or b <= 1:
            return np.nan
        else:
            t_expected = self.t_mean(a, b)
            tsq_expected = 0.25 * (a + b) * ((a - b) ** 2 + (a - 1) + (b - 1)) / ((a - 1) * (b - 1))
            return np.sqrt(tsq_expected - t_expected ** 2)


    def p_of_y_given_yhat(self,
                          y,
                          y_mode):
                          # y_scale,
                          # a,
                          # b):

        # t = self.t_mode(a, b) + (y - y_mode) / y_scale
        # return self.f(t, a, b) / y_scale

        y_scale = np.exp(self.log_scale)
        a = np.exp(self.log_a)
        b = np.exp(self.log_b)

        t = self.t_mode(a, b) + (y - y_mode) / y_scale
        return self.f(t, a, b) / y_scale


    def p_of_y_given_phi(self,
                         y,
                         phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)


    def p_mean_std(self, y_mode, y_scale, a, b):
        y_mean = self.t_mean(a, b) * y_scale + y_mode
        y_std = self.t_std(a, b) * y_scale
        return y_mean, y_std


    def t_quantile(self, q, a, b):
        x_q = BetaIncInv(a, b, q)
        t_q = (2 * x_q - 1) * np.sqrt(a + b) / np.sqrt(1 - (2 * x_q - 1) ** 2)
        return t_q


    def y_quantile(self, q, y_hat, s, a, b):
        t_q = self.t_quantile(q, a, b)
        y_q = (t_q - self.t_mode(a, b)) * s + y_hat
        return y_q


    def estimate_predictive_information(self,
                                        y,
                                        yhat,
                                        y_scale,
                                        a,
                                        b):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))
        N = len(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat, y_scale, a, b).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))

        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)

        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)
        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


@handle_errors
class GaussianNoiseModel:
    """
    Class used to obtain +/- sigma from the Gaussian noise model
    in the GE model. The sigma, which is a function of y_hat, can
    be used to plot confidence intervals around y_hat.


    model: (mavenn.Model object)
         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Gaussian noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the confidence intervals will be computed.
        This should be the output of the GE model.

    """


    def __init__(self,
                 model,
                 yhat_GE,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat_GE = yhat_GE

        self.polynomial_weights = self.model.get_nn().layers[8].get_weights()[0].copy()
        logsigma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            logsigma += self.polynomial_weights[polynomial_index][0] * np.power(yhat_GE, polynomial_index)

        # this is sigma(y)
        self.sigma = np.exp(logsigma)

        if q is not None:
            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(yhat_GE+self.sigma*np.sqrt(2)*erfinv(2*current_q-1))

            #self.user_quantile_values = np.array(self.user_quantile_values).reshape(len(yhat_GE), len(q))


    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # recompute logsimga here instead of using self.gamma since y,yhat input
        # to this method could be different than init
        logsigma = 0

        for polynomial_index in range(len(self.polynomial_weights)):
            logsigma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        sigma = np.exp(logsigma)

        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((y - yhat) ** 2) / (2 * sigma ** 2))


    def p_of_y_given_phi(self,
                       y,
                       phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        print(y_hat_of_phi.shape)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)

    # TODO:
    '''
    arguments should be x, y. 
    1) map x to phi
    2) p_y_given_phi
    2) then compute mutual information via 
    '''
    def estimate_predictive_information(self,
                                        y,
                                        yhat):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat)))
        N = len(np.log2(self.p_of_y_given_yhat(y, yhat)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))

        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)
        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)

        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


@handle_errors
class CauchyNoiseModel:

    """
    Class used to obtain +/- sigma from the Cauchy noise model
    in the GE model. The sigma, which is a function of y_hat, can
    be used to plot confidence intervals around y_hat.


    model: (mavenn.Model object)
         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Cauchy noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the confidence intervals will be computed.
        This should be the output of the GE model.

    """

    def __init__(self,
                 model,
                 yhat,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat = yhat
        self.q = q

        self.polynomial_weights = self.model.get_nn().layers[8].get_weights()[0].copy()

        self.log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            self.log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        self.plus_sigma_quantile = self.y_quantile(0.16, self.yhat)
        self.minus_sigma_quantile = self.y_quantile(0.84, self.yhat)

        if q is not None:
            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(self.y_quantile(current_q, self.yhat).ravel())


    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # recompute gamma here instead of using self.gamma since y,yhat input
        # to this method could be different than init
        log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        return cauchy(loc=yhat, scale=np.exp(log_gamma)).pdf(y)


    def p_of_y_given_phi(self,
                         y,
                         phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)


    def y_quantile(self,
                   user_quantile,
                   yhat):

        """
        user_quantile: (float between [0,1])
            The value representing the quantile which will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # compute gamma for the yhat entered
        log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        return cauchy(loc=yhat, scale=np.exp(log_gamma)).ppf(user_quantile)


    def estimate_predictive_information(self,
                                        y,
                                        yhat):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat)))

        N = len(np.log2(self.p_of_y_given_yhat(y, yhat)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))
        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)

        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)

        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


@handle_errors
def load(filename):

        """
        Method that will load a mave-nn model

        parameters
        ----------
        filename: (str)
            filename of saved model.

        returns
        -------
        loaded_model (mavenn-Model object)
            The model object that can be used to make predictions etc.

        """

        load_config = pd.read_csv(filename + '.csv', index_col=[0])

        # validate load config...
        check(len(load_config) >= 1, 'Length of loaded model file must be at least 1.')

        # get regression_type variable to determine whether GE or MPA regression.
        regression_type = load_config['regression_type'].values[0]

        if regression_type=='GE':

            # load configuration file
            load_config = pd.read_csv(filename+'.csv', index_col=[0])

            # convert it to a dictionary
            #config_dict = load_config[load_config.columns[2:]].loc[0].to_dict()
            config_dict = load_config.loc[0].to_dict()

            x_train = load_config['x'].values
            y_train = load_config['y'].values

            diffeomorphic_mean = load_config['diffeomorphic_mean'].values
            diffeomorphic_std = load_config['diffeomorphic_std'].values

            # delete keys not required for model loading
            config_dict.pop('x', None)
            config_dict.pop('y', None)
            config_dict.pop('L', None)
            config_dict.pop('model', None)
            config_dict.pop('diffeomorphic_mean', None)
            config_dict.pop('diffeomorphic_std', None)
            config_dict.pop('define_model', None)
            config_dict.pop('learning_rate', None)

            # create object for loaded model
            loaded_model = mavenn.Model(x=x_train,
                                        y=y_train,
                                        **config_dict)

            # this gauge fixing method merely sets up the correct
            # architecture for the neural network, for which the
            # weights can then be set using a trained model
            loaded_model.gauge_fix_model(load_model=True,
                                         diffeomorphic_mean=diffeomorphic_mean[0],
                                         diffeomorphic_std=diffeomorphic_std[0])

            # set weights using a trained model.
            loaded_model.get_nn().load_weights(filename+'.h5')

            return loaded_model
        else:

            load_config = pd.read_csv(filename + '.csv', index_col=[0], keep_default_na=False)

            # single_y = load_config['y'].values[0]
            # single_y = single_y[1:len(single_y)-2].split('.')
            # single_y = [int(i) for i in single_y]
            # single_y = np.array(single_y)

            single_y = load_config['y'].values[0]
            single_y = single_y[1:len(single_y) - 1].split(' ')
            single_y = [int(i) for i in single_y]
            single_y = np.array(single_y)

            single_x = load_config['x'].values

            single_ct_n = load_config['ct_n'][0]
            single_ct_n = [int(float(single_ct_n[1:len(single_ct_n) - 1]))]

            diffeomorphic_mean = load_config['diffeomorphic_mean'].values
            diffeomorphic_std = load_config['diffeomorphic_std'].values

            config_dict = load_config.loc[0].to_dict()

            config_dict.pop('x', None)
            config_dict.pop('y', None)
            config_dict.pop('ct_n', None)
            config_dict.pop('model', None)
            config_dict.pop('L', None)
            config_dict.pop('diffeomorphic_mean', None)
            config_dict.pop('diffeomorphic_std', None)
            config_dict.pop('define_model', None)
            config_dict.pop('learning_rate', None)

            # create object for loaded model
            loaded_model = mavenn.Model(x=single_x,
                                        y=single_y,
                                        ct_n=single_ct_n,
                                        **config_dict)

            # this gauge fixing method merely sets up the correct
            # architecture for the neural network, for which the
            # weights can then be set using a trained model
            loaded_model.gauge_fix_model(load_model=True,
                                         diffeomorphic_mean=diffeomorphic_mean[0],
                                         diffeomorphic_std=diffeomorphic_std[0])

            # set weights using a trained model.
            loaded_model.get_nn().load_weights(filename + '.h5')

            return loaded_model


@handle_errors
def vec_data_to_mat_data(y_n,
                         ct_n=None,
                         x_n=None):
    """

    Function to transform from vector to matrix format for MPA
    regression and MPA model evaluation.

    parameters
    ----------

    y_n: (array-like of ints)
        List of N bin numbers y. Must be set by user.

    ct_n: (array-like of ints)
        List N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (array-like)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    returs
    ------

    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.
    """

    # Cast y as array of ints
    y_n = np.array(y_n).astype(int)
    N = len(x_n)

    # Cast x as array and get length
    if x_n is None:
        x_n = np.arange(N)
    else:
       x_n = np.array(x_n)
       #assert len(x_n) == N, f'len(y_n)={len(y_n)} and len(x_n)={N} do not match.'

    # Get ct
    if ct_n is None:
        ct_n = np.ones(N).astype(int)
    #else:
        #assert len(ct_n) == N, f'len(ct_n)={len(ct_n)} and len(x_n)={N} do not match.'

    # This case is only for loading data. Should be tested/made more robust
    if N == 1:
        # should do check like check(mavenn.load_model==True,...

        return y_n.reshape(-1, y_n.shape[0]), x_n

    # Create dataframe
    data_df = pd.DataFrame()
    data_df['x'] = x_n
    data_df['y'] = y_n
    data_df['ct'] = ct_n

    # Sum over repeats
    data_df = data_df.groupby(['x','y']).sum().reset_index()

    # Pivot dataframe
    data_df = data_df.pivot(index='x', columns='y', values='ct')
    data_df = data_df.fillna(0).astype(int)

    # Clean dataframe
    data_df.reset_index(inplace=True)
    data_df.columns.name = None

    # Get ct_my values
    cols = [c for c in data_df.columns if not c in ['x']]

    ct_my = data_df[cols].values.astype(int)

    # Get x_m values
    x_m = data_df['x'].values

    return ct_my, x_m
