from mavenn.src.validate import validate_input
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import onehot_encode_array, \
    _generate_nbr_features_from_sequences, _generate_all_pair_features_from_sequences

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Lambda, Concatenate
from tensorflow.keras.constraints import non_neg as nonneg
from mavenn.src.likelihood_layers import *

@handle_errors
class GlobalEpistasisModel:

    """
    Class that implements global epistasis regression.

    attributes
    ----------

    X: (array-like of str)
        Sequence inputs; can represent DNA, RNA, or protein sequences, as
        specified by the alphabet attribute.

    y: (array-like of float)
        Measurement values corresponding to the sequences in X.

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation.

    monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    custom_architecture: (tf.model)
        Specify a custom neural network architecture (including both the
        G-P map and the measurement process) to fit to data.

    ohe_single_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.


    polynomial_order_ll: (int)
        Order of polynomial which specifies the dependence of the noise-model
        distribution paramters, used in the computation of likelihood, on yhat.
        (Only used for GE regression).

    """

    def __init__(self,
                 X,
                 y,
                 gpmap_type,
                 test_size,
                 alphabet,
                 monotonic,
                 custom_architecture,
                 ohe_single_batch_size,
                 polynomial_order_ll):

        # set class attributes
        self.X, self.y = X, y
        self.gpmap_type = gpmap_type
        self.test_size = test_size
        self.monotonic = monotonic
        self.alphabet = alphabet
        self.custom_architecture = custom_architecture
        self.ohe_single_batch_size = ohe_single_batch_size
        self.polynomial_order_ll = polynomial_order_ll

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (gpmap and measurement)
        # and are set after the model has been fit to data.
        self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # perform input checks to validate attributes
        self._input_checks()

        # clarify that X and y are the training datasets (including validation sets)
        self.x_train, self.y_train = self.X, self.y

        # set characters
        if self.alphabet == 'dna':
            self.characters = ['A', 'C', 'G', 'T']
        elif self.alphabet == 'rna':
            self.characters = ['A', 'C', 'G', 'U']
        elif self.alphabet == 'protein':
            self.characters = ['A', 'C', 'D', 'E', 'F',
                               'G', 'H', 'I', 'K', 'L',
                               'M', 'N', 'P', 'Q', 'R',
                               'S', 'T', 'V', 'W', 'Y']

        # compute appropriate one-hot encoding of sequences
        if gpmap_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

        elif gpmap_type == 'neighbor':
            # one-hot encode sequences in batches in a vectorized way
            # TODO: vectorize neighbor feature creation
            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

            # Generate neighbor one-hot encoding.
            X_train_neighbor = _generate_nbr_features_from_sequences(self.x_train, self.alphabet)

            # Append additive and pairwise features together.
            X_train_features = np.hstack((X_train_additive, X_train_neighbor))

            # this is the input to the neighbor model
            self.input_seqs_ohe = X_train_features

        elif gpmap_type == 'pairwise':
            # one-hot encode sequences in batches in a vectorized way
            # TODO: vectorize pairwise feature creation
            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

            # Generate pairwise one-hot encoding.
            X_train_pairwise = _generate_all_pair_features_from_sequences(self.x_train, self.alphabet)

            # Append additive and pairwise features together.
            X_train_features = np.hstack((X_train_additive, X_train_pairwise))

            # this is the input to the pairwise model
            self.input_seqs_ohe = X_train_features

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train).reshape(np.shape(self.y_train)[0], 1)

    # CONTINE CODE REVIEW BELOW (date: 20.07.20)

    def _input_checks(self):

        """
        Validate parameters passed to the GlobalEpistasis constructor
        """
        # validate input df
        #self.df = validate_input(self.df)

        check(isinstance(self.X, (list, np.ndarray)),
              'type(X) = %s must be of type list or np.array' % type(self.X))
        self.X = np.array(self.X)

        check(isinstance(self.y, (list, np.ndarray)),
              'type(y) = %s must be of type list or np.array' % type(self.y))
        self.y = np.array(self.y)

        check(len(self.X) == len(self.y),
              'length of inputs (X, y) must be equal')
        self.num_measurements = len(self.X)

        # check that gpmap_type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              'gpmap = %s; must be "additive", "neighbor", or "pairwise"' %
              self.gpmap_type)

    def define_model(self,
                     noise_model,
                     num_nodes_hidden_measurement_layer=50,
                     custom_architecture=None):

        """
        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API. If custom_architecture is not None,
        this is used instead as the model architecture.

        parameters
        ----------
        noise_model: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        num_nodes_hidden_measurement_layer: (int)
            Number of nodes to use in the hidden layer of the measurement network
            of the GE model architecture.

        custom_architecture: (tf.model)
            A custom neural network architecture that replaces the entire
            default architecture implemented. Set to None to use default.

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        # check that noise_model valid
        check(noise_model in {'Gaussian', 'Cauchy', 'SkewedT'},
              'noise_model = %s; must be "Gaussian", "Cauchy", or "SkewedT"' %
              noise_model)

        # If user has not provided custom architecture, implement a default architecture
        if custom_architecture is None:

            number_input_layer_nodes = len(self.input_seqs_ohe[0])+1
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                    output_shape=((len(self.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
            labels_input = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + 1],
                                  output_shape=((1, )), trainable=False, name='Labels_input')(inputTensor)

            phi = Dense(1, name='phi')(sequence_input)

            # implement monotonicity constraints
            if self.monotonic:
                intermediateTensor = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid',
                                           kernel_constraint=nonneg())(phi)
                yhat = Dense(1, kernel_constraint=nonneg(),name='y_hat')(intermediateTensor)

                concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])

                # dynamic likelihood class instantiation by the globals dictionary
                # manual instantiation can be done as follows:
                # outputTensor = GaussianLikelihoodLayer()(concatenateLayer)

                likelihoodClass = globals()[noise_model + 'LikelihoodLayer']
                outputTensor = likelihoodClass(self.polynomial_order_ll)(concatenateLayer)

            else:
                intermediateTensor = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
                yhat = Dense(1, name='y_hat')(intermediateTensor)

                concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
                likelihoodClass = globals()[noise_model + 'LikelihoodLayer']
                outputTensor = likelihoodClass(self.polynomial_order_ll)(concatenateLayer)

            # create the model:
            model = Model(inputTensor, outputTensor)
            self.model = model
            self.num_nodes_hidden_measurement_layer = num_nodes_hidden_measurement_layer

            return model

        # if user has provided custom architecture
        else:
            self.model = custom_architecture
            return custom_architecture

    # Do code review below: 20.07.22
    def ge_nonlinearity(self,
                        phi):

        """
        Compute the GE nonlinearity at specified values of phi.

        parameters
        ----------

        phi: (float or array-like)
            Latent phenotype value(s) on which the GE nonlinearity
            wil be evaluated.

        returns
        -------
        y_hat: (float or array-like)
            The nonlinear GE function evaluated at phi.

        """

        # TODO disable method if custom_architecture is specified.

        ge_model_input = Input((1,))
        next_input = ge_model_input

        # the following variable is the index of
        phi_index = 4
        yhat_index = 7

        # Form model using functional API in a loop, starting from
        # phi input, and ending on network output
        for layer in self.model.layers[phi_index:yhat_index]:
            next_input = layer(next_input)

        # Form gauge fixed GE_nonlinearity model
        ge_model = Model(inputs=ge_model_input, outputs=next_input)

        # compute the value of the nonlinearity for a given phi
        y_hat = ge_model.predict([phi])

        return y_hat


class NoiseAgnosticModel:

    """
    Class that implements Noise agnostic regression.


    attributes
    ----------

    X: (array-like)
        Input pandas DataFrame containing sequences. X are
        DNA, RNA, or protein sequences to be regressed over

    y: (array-like)
        y represents counts in bins corresponding to the sequences X

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    custom_architecture: (tf.model)
        Specify a custom neural network architecture (including both the
        G-P map and the measurement process) to fit to data.

    ohe_single_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.
    """

    def __init__(self,
                 X,
                 y,
                 gpmap_type,
                 test_size,
                 alphabet,
                 custom_architecture,
                 ohe_single_batch_size):

        # set class attributes
        self.X = X
        self.y = y
        self.gpmap_type = gpmap_type
        self.test_size = test_size
        self.alphabet = alphabet
        self.custom_architecture = custom_architecture
        self.ohe_single_batch_size = ohe_single_batch_size

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (gpmap and measurement)
        # and are set after the model has been fit to data.
        self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.na_model = None

        # perform input checks to validate attributes
        self._input_checks()

        self.x_train, self.y_train = self.X, self.y

        if self.alphabet == 'dna':
            self.characters = ['A', 'C', 'G', 'T']
        elif self.alphabet == 'rna':
            self.characters = ['A', 'C', 'G', 'U']
        elif self.alphabet == 'protein':
            self.characters = ['A', 'C', 'D', 'E', 'F',
                               'G', 'H', 'I', 'K', 'L',
                               'M', 'N', 'P', 'Q', 'R',
                               'S', 'T', 'V', 'W', 'Y']

        if gpmap_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

        elif gpmap_type == 'neighbor':

            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

            # Generate pairwise one-hot encoding.
            X_train_neighbor = _generate_nbr_features_from_sequences(self.x_train, self.alphabet)

            # Append additive and pairwise features together.
            X_train_features = np.hstack((X_train_additive, X_train_neighbor))

            # this is the input to the neighbor model
            self.input_seqs_ohe = X_train_features

        elif gpmap_type == 'pairwise':

            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_single_batch_size)

            # Generate pairwise one-hot encoding.
            X_train_pairwise = _generate_all_pair_features_from_sequences(self.x_train, self.alphabet)

            # Append additive and pairwise features together.
            X_train_features = np.hstack((X_train_additive, X_train_pairwise))

            # this is the input to the pairwise model
            self.input_seqs_ohe = X_train_features

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train)

    def _input_checks(self):

        """
        Validate parameters passed to the NoiseAgnosticRegression constructor
        """
        check(isinstance(self.X, (list, np.ndarray)),
              'type(X) = %s must be of type list or np.array' % type(self.X))

        check(isinstance(self.y, (list, np.ndarray)),
              'type(y) = %s must be of type list or np.array' % type(self.y))

        check(len(self.X) == len(self.y),
              'length of inputs (X, y) must be equal')

        # check that model type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              'model_type = %s; must be "additive", "neighbor", or "pairwise"' %
              self.gpmap_type)

    def define_model(self,
                     num_nodes_hidden_measurement_layer=10,
                     custom_architecture=None):

        """
        Defines the architecture of the noise agnostic regression model.
        using the tensorflow.keras functional API. If custom_architecture is not None,
        this is used instead as the model architecture.

        parameters
        ----------
        num_nodes_hidden_measurement_layer: (int)
            Number of nodes to use in the hidden layer of the measurement network
            of the GE model architecture.

        custom_architecture: (tf.model)
            A custom neural network architecture that replaces the entire
            default architecture implemented. Set to None to use default.

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        if custom_architecture is None:

            number_input_layer_nodes = len(self.input_seqs_ohe[0])+self.y.shape[1]

            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                    output_shape=((len(self.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
            labels_input = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + self.y.shape[1]],
                                  output_shape=((1, )), trainable=False, name='Labels_input')(inputTensor)

            phi = Dense(1, use_bias=True, name='phi')(sequence_input)

            intermediateTensor = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
            yhat = Dense(np.shape(self.y_train[0])[0], name='yhat', activation='softmax')(intermediateTensor)

            concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
            outputTensor = NALikelihoodLayer(number_bins=np.shape(self.y_train[0])[0])(concatenateLayer)

            #create the model:
            model = Model(inputTensor, outputTensor)
            self.model = model
            self.num_nodes_hidden_measurement_layer = num_nodes_hidden_measurement_layer
            return model
        else:
            self.model = custom_architecture
            return custom_architecture

    def noise_model(self,
                    phi):

        """
        Method used to evaluate NA noise model.

        parameters
        ----------

        phi: (float)
            Latent phenotype value(s) on which the NA noise model
            wil be evaluated.

        returns
        -------
        pi: (array-like)
            The nonlinear NA function evaluated for input phi.

        """
        # TODO disable method if custom_architecture is specified.

        na_model_input = Input((1,))
        next_input = na_model_input

        # the following variable is the index of
        phi_index = 4
        yhat_index = 7

        # Form model using functional API in a loop, starting from
        # phi input, and ending on network output
        for layer in self.model.layers[phi_index:yhat_index]:
            next_input = layer(next_input)

        # Form gauge fixed GE_nonlinearity model
        na_model = Model(inputs=na_model_input, outputs=next_input)

        # compute the value of the nonlinearity for a given phi
        y_hat = na_model.predict([phi])

        return y_hat
