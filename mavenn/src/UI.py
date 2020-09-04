from mavenn.src.validate import validate_input
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import vec_data_to_mat_data
from mavenn.src.utils import onehot_encode_array, \
    _generate_nbr_features_from_sequences, _generate_all_pair_features_from_sequences

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Lambda, Concatenate
from tensorflow.keras.constraints import non_neg as nonneg
from mavenn.src.likelihood_layers import *

import numbers

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

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model parameters
        dependent on y_hat, and thus render the noise model heteroskedastic. Set
        to zero for a homoskedastic noise model. (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters $\theta$.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters $\eta$.

    """

    def __init__(self,
                 X,
                 y,
                 alphabet,
                 gpmap_type,
                 ge_nonlinearity_monotonic,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization):

        # set class attributes
        self.X, self.y = X, y
        self.gpmap_type = gpmap_type
        self.alphabet = alphabet
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # perform input checks to validate attributes
        self._input_checks()

        # clarify that X and y are the training datasets (including validation sets)
        self.x_train, self.y_train = self.X, self.y

        # record sequence length for convenience
        self.L = len(self.x_train[0])

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
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

        elif gpmap_type == 'neighbor':
            # one-hot encode sequences in batches in a vectorized way
            # TODO: vectorize neighbor feature creation
            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

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
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

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

        # useful tuple to check if some value is a number

        # validate input df
        #self.df = validate_input(self.df)

        check(isinstance(self.X, (list, np.ndarray)),
              'type(X) = %s must be of type list or np.array' % type(self.X))
        self.X = np.array(self.X)

        check(isinstance(self.X[0], str),
              'type(x_train) = %s must be of type str' % type(self.X))

        check(isinstance(self.y, (list, np.ndarray)),
              'type(y) = %s must be of type list or np.array' % type(self.y))
        self.y = np.array(self.y)

        check(len(self.X) == len(self.y),
              'length of inputs (X, y) must be equal')
        self.num_measurements = len(self.X)

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic, (bool, np.bool, np.bool_)), 'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral), 'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0, 'ge_heteroskedasticity_order must be >= 0')

        # check that gpmap_type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              'gpmap_type = %s; must be "additive", "neighbor", or "pairwise"' %
              self.gpmap_type)

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real), 'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0, 'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real), 'eta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0, 'eta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral), 'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0, 'ohe_batch_size must be > 0')

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):

        """
        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT'},
              'p_of_all_y_given_phi = %s; must be "Gaussian", "Cauchy", or "SkewedT"' %
              ge_noise_model_type)

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral), 'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0, 'ge_nonlinearity_hidden_nodes must be greater than 0.')

        number_input_layer_nodes = len(self.input_seqs_ohe[0])+1
        inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                output_shape=((len(self.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
        labels_input = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + 1],
                              output_shape=((1, )), trainable=False, name='Labels_input')(inputTensor)

        phi = Dense(1, name='phi',
                    kernel_regularizer=tf.keras.regularizers.l2(self.theta_regularization))(sequence_input)

        # implement monotonicity constraints
        if self.ge_nonlinearity_monotonic==True:

            intermediateTensor = Dense(ge_nonlinearity_hidden_nodes, activation='sigmoid',
                                       kernel_constraint=nonneg())(phi)
            yhat = Dense(1, kernel_constraint=nonneg(),name='y_hat')(intermediateTensor)

            concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])

            # dynamic likelihood class instantiation by the globals dictionary
            # manual instantiation can be done as follows:
            # outputTensor = GaussianLikelihoodLayer()(concatenateLayer)

            likelihoodClass = globals()[ge_noise_model_type + 'LikelihoodLayer']
            outputTensor = likelihoodClass(self.ge_heteroskedasticity_order)(concatenateLayer)

        else:

            intermediateTensor = Dense(ge_nonlinearity_hidden_nodes, activation='sigmoid')(phi)
            yhat = Dense(1, name='y_hat')(intermediateTensor)

            concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
            likelihoodClass = globals()[ge_noise_model_type + 'LikelihoodLayer']
            outputTensor = likelihoodClass(self.ge_heteroskedasticity_order, self.eta_regularization)(concatenateLayer)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model

    # Do code review below: 20.07.22
    def phi_to_yhat(self,
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

        ge_model_input = Input((1,))
        next_input = ge_model_input

        # the following variable is the index of
        # phi_index = 4
        # yhat_index = 7
        phi_index = 3
        yhat_index = 6

        # Form model using functional API in a loop, starting from
        # phi input, and ending on network output
        for layer in self.model.layers[phi_index:yhat_index]:
            next_input = layer(next_input)

        # Form gauge fixed GE_nonlinearity model
        ge_model = Model(inputs=ge_model_input, outputs=next_input)

        # compute the value of the nonlinearity for a given phi
        y_hat = ge_model.predict([phi])

        return y_hat

@handle_errors
class GlobalEpistasisModelMultipleReplicates:

    """
    Class that implements global epistasis regression with multiple targets
    for each sequence. Missing target values/NANs are allowed.

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
        # gauge fixing the neural network model (x_to_phi and measurement)
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
        # self.y_train = np.array(self.y_train).reshape(np.shape(self.y_train)[0], 1)

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

        # check(len(self.X) == len(self.y),
        #       'length of inputs (X, y) must be equal')
        # self.num_measurements = len(self.X)

        # check that gpmap_type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              'x_to_phi = %s; must be "additive", "neighbor", or "pairwise"' %
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

        # check that p_of_all_y_given_phi valid
        check(noise_model in {'Gaussian', 'Cauchy', 'SkewedT'},
              'p_of_all_y_given_phi = %s; must be "Gaussian", "Cauchy", or "SkewedT"' %
              noise_model)

        # If user has not provided custom architecture, implement a default architecture
        if custom_architecture is None:

            if len(self.y_train.shape) == 1:
                number_of_replicate_targets = 1
            else:
                number_of_replicate_targets = min(self.y_train.shape)

            print('number of y nodes to add: ',number_of_replicate_targets)
            # create input layer with nodes allowing sequence to be input and also
            # target labels to be input, together.
            #number_input_layer_nodes = len(self.input_seqs_ohe[0])+self.y_train.shape[0]
            number_input_layer_nodes = len(self.input_seqs_ohe[0]) + number_of_replicate_targets
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                    output_shape=((len(self.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)

            replicates_input = []

            #number_of_replicate_targets = self.y_train.shape[0]

            for replicate_layer_index in range(number_of_replicate_targets):

                # build up lambda layers, on step at a time, which will be
                # fed to each of the measurement blocks
                print(replicate_layer_index, replicate_layer_index + 1)

                temp_replicate_layer = Lambda(lambda x:
                                              x[:, len(self.input_seqs_ohe[0])+replicate_layer_index:
                                              len(self.input_seqs_ohe[0]) + replicate_layer_index + 1],
                                              output_shape=((1,)), trainable=False,
                                              name='Labels_input_'+str(replicate_layer_index))(inputTensor)

                replicates_input.append(temp_replicate_layer)

            # labels_input_rep1 = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + 1],
            #                       output_shape=((1, )), trainable=False, name='Labels_input_1')(inputTensor)
            #
            # labels_input_rep2 = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0])+1:len(self.input_seqs_ohe[0]) + 2],
            #                            output_shape=((1,)), trainable=False, name='Labels_input_2')(inputTensor)

            # sequence to latent phenotype
            phi = Dense(1, name='phi')(sequence_input)

            # implement monotonicity constraints
            if self.monotonic:

                # phi feeds into each of the replicate intermediate layers
                intermediate_layers = []
                for intermediate_index in range(number_of_replicate_targets):

                    temp_intermediate_layer = Dense(num_nodes_hidden_measurement_layer,
                                                    activation='sigmoid',
                                                    kernel_constraint=nonneg(),
                                                    name='intermediate_bbox_'+str(intermediate_index))(phi)

                    intermediate_layers.append(temp_intermediate_layer)

                # intermediateTensor_1 = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid',
                #                            kernel_constraint=nonneg(), name='intermediate_bbox_1')(phi)
                #
                # intermediateTensor_2 = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid',
                #                              kernel_constraint=nonneg(), name='intermediate_bbox_2')(phi)

                # build up yhat layers, going from phi through an intermediate,
                # nonlinearly activated layer, to each of the y_hat nodes

                # y_hat, each representing a prediction for a replicate y
                yhat_layers = []
                for yhat_index in range(number_of_replicate_targets):

                    temp_yhat_layer = Dense(1, kernel_constraint=nonneg(),
                                            name='y_hat_rep_'+str(yhat_index))(intermediate_layers[yhat_index])
                    yhat_layers.append(temp_yhat_layer)

                #yhat_rep1 = Dense(1, kernel_constraint=nonneg(),name='y_hat_rep1')(intermediateTensor_1)
                #yhat_rep2 = Dense(1, kernel_constraint=nonneg(), name='y_hat_rep2')(intermediateTensor_2)

                # concatenate yhat_ith with labels_input_rep_ith into the likelihood layers
                # to compute the loss

                concatenateLayer_rep_input = []

                for concat_index in range(number_of_replicate_targets):

                    temp_concat = Concatenate(name='yhat_and_rep_'+str(concat_index))\
                        ([yhat_layers[concat_index], replicates_input[concat_index]])

                    concatenateLayer_rep_input.append(temp_concat)

                #concatenateLayer_rep1 = Concatenate(name='yhat_and_rep1')([yhat_rep1, labels_input_rep1])
                #concatenateLayer_rep2 = Concatenate(name='yhat_and_rep2')([yhat_rep2, labels_input_rep2])

                # dynamic likelihood class instantiation by the globals dictionary
                # manual instantiation can be done as follows:
                # outputTensor = GaussianLikelihoodLayer()(concatenateLayer)

                likelihoodClass = globals()[noise_model + 'LikelihoodLayer']

                #ll_rep1 = likelihoodClass(self.polynomial_order_ll)(concatenateLayer_rep1)
                #ll_rep2 = likelihoodClass(self.polynomial_order_ll)(concatenateLayer_rep2)

                ll_rep_layers = []
                for ll_index in range(number_of_replicate_targets):
                    temp_ll_layer = likelihoodClass(self.polynomial_order_ll)(concatenateLayer_rep_input[ll_index])
                    ll_rep_layers.append(temp_ll_layer)


                #outputTensor = [ll_rep1, ll_rep2]
                outputTensor = ll_rep_layers


            else:
                # TODO: implement this without monotonicity constraints
                pass
                # intermediateTensor = Dense(num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
                # yhat = Dense(1, name='y_hat')(intermediateTensor)
                #
                # concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
                # likelihoodClass = globals()[p_of_all_y_given_phi + 'LikelihoodLayer']
                # outputTensor = likelihoodClass(self.polynomial_order_ll)(concatenateLayer)

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
        y_hat = ge_model.x_to_yhat([phi])

        return y_hat


class MeasurementProcessAgnosticModel:

    """
    Class that implements Measurement process agnostic regression.


    attributes
    ----------

    x: (array-like)
        Input pandas DataFrame containing sequences. X are
        DNA, RNA, or protein sequences to be regressed over

    y: (array-like)
        y represents counts in bins corresponding to the sequences X

    ct_n: (array-like of ints)
        List N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters $\theta$.
    """

    def __init__(self,
                 x,
                 y,
                 ct_n,
                 gpmap_type,
                 alphabet,
                 theta_regularization,
                 ohe_batch_size):

        # set class attributes
        self.x = x
        self.y = y
        self.ct_n = ct_n
        self.gpmap_type = gpmap_type
        self.alphabet = alphabet
        self.theta_regularization = theta_regularization
        self.ohe_batch_size = ohe_batch_size

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        self.na_hidden_nodes = None
        self.theta_gf = None
        self.na_model = None

        # perform input checks to validate attributes
        self._input_checks()

        self.y, self.x = vec_data_to_mat_data(x_n=x,
                                              y_n=y,
                                              ct_n=ct_n)

        # record sequence length for convenience
        self.L = len(self.x[0])

        # Record number of bins
        self.Y = self.y.shape[1]
        self.all_y = np.arange(self.Y).astype(int)

        self.x_train, self.y_train = self.x, self.y

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
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

        elif gpmap_type == 'neighbor':

            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

            # Generate pairwise one-hot encoding.
            X_train_neighbor = _generate_nbr_features_from_sequences(self.x_train, self.alphabet)

            # Append additive and pairwise features together.
            X_train_features = np.hstack((X_train_additive, X_train_neighbor))

            # this is the input to the neighbor model
            self.input_seqs_ohe = X_train_features

        elif gpmap_type == 'pairwise':

            # Generate additive one-hot encoding.
            X_train_additive = onehot_encode_array(self.x_train, self.characters, self.ohe_batch_size)

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

        # useful tuple to check if some value is a number

        check(isinstance(self.x, (list, np.ndarray, pd.DataFrame, pd.Series)),
              'type(X) = %s must be of type list or np.array' % type(self.x))

        #check(isinstance(self.x[0], str),
        #      'type(x_train) = %s must be of type str' % type(self.x))

        check(isinstance(self.y, (list, np.ndarray, pd.DataFrame, pd.Series)),
              'type(y) = %s must be of type list or np.array' % type(self.y))

        check(isinstance(self.ct_n, (list, np.ndarray, pd.DataFrame, pd.Series)),
              'type(ct_n) = %s must be of type list or np.array' % type(self.ct_n))

        # check(len(self.x) == len(self.y),
        #      'length of inputs (X, y) must be equal')

        # check that model type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              'model_type = %s; must be "additive", "neighbor", or "pairwise"' %
              self.gpmap_type)

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real), 'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0, 'theta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral), 'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0, 'ohe_batch_size must be > 0')

    def define_model(self,
                     na_hidden_nodes=10):

        """
        Defines the architecture of the noise agnostic regression model.
        using the tensorflow.keras functional API. If custom_architecture is not None,
        this is used instead as the model architecture.

        parameters
        ----------
        na_hidden_nodes: (int)
            Number of nodes to use in the hidden layer of the measurement network
            of the GE model architecture.

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        # useful tuple to check if some value is a number

        check(isinstance(na_hidden_nodes, numbers.Integral), 'na_hidden_nodes must be a number.')

        check(na_hidden_nodes > 0, 'na_hidden_nodes must be greater than 0.')

        number_input_layer_nodes = len(self.input_seqs_ohe[0])+self.y.shape[1]

        inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                output_shape=((len(self.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
        labels_input = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + self.y.shape[1]],
                              output_shape=((1, )), trainable=False, name='Labels_input')(inputTensor)

        phi = Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(self.theta_regularization),
                    use_bias=True, name='phi')(sequence_input)

        intermediateTensor = Dense(na_hidden_nodes, activation='sigmoid')(phi)
        yhat = Dense(np.shape(self.y_train[0])[0], name='yhat', activation='softmax')(intermediateTensor)

        concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
        outputTensor = MPALikelihoodLayer(number_bins=np.shape(self.y_train[0])[0])(concatenateLayer)

        #create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.na_hidden_nodes = na_hidden_nodes
        return model

    def p_of_all_y_given_phi(self,
                             phi):

        """
        Method used to evaluate MPA noise model.

        parameters
        ----------

        phi: (float)
            Latent phenotype value(s) on which the MPA noise model
            wil be evaluated.

        returns
        -------
        pi: (array-like)
            The nonlinear MPA function evaluated for input phi.

        """

        na_model_input = Input((1,))
        next_input = na_model_input

        # the following variable is the index of
        phi_index = 3
        yhat_index = 6

        # Form model using functional API in a loop, starting from
        # phi input, and ending on network output
        for layer in self.model.layers[phi_index:yhat_index]:
            next_input = layer(next_input)

        # Form gauge fixed GE_nonlinearity model
        na_model = Model(inputs=na_model_input, outputs=next_input)

        # compute the value of the nonlinearity for a given phi

        p_of_dot_given_phi = na_model.predict([phi])

        return p_of_dot_given_phi
