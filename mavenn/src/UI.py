import numpy as np
import pandas as pd
import numbers
import time

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers \
    import Dense, Activation, Input, Lambda, Concatenate

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import vec_data_to_mat_data
from mavenn.src.likelihood_layers import *  #TODO: List specific imports instead
from mavenn.src.dev import x_to_features
from mavenn.src.validate import validate_alphabet
from mavenn.src.dev import GlobalEpistasisLayer

@handle_errors
class GlobalEpistasisModel:

    """
    Class that implements global epistasis regression.

    attributes
    ----------

    x: (array-like of str)
        Sequence inputs; can represent DNA, RNA, or protein sequences, as
        specified by the alphabet attribute.

    y: (array-like of float)
        Measurement values corresponding to the sequences in X.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

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

    @handle_errors
    def __init__(self,
                 x,
                 y,
                 alphabet,
                 gpmap_type,
                 ge_nonlinearity_monotonic,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization):

        # set class attributes
        self.x, self.y = x, y
        self.gpmap_type = gpmap_type
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
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

        check(isinstance(self.x, (list, np.ndarray)),
              'type(X) = %s must be of type list or np.array' % type(self.x))
        self.x = np.array(self.x)

        check(isinstance(self.x[0], str),
              'type(x_train) = %s must be of type str' % type(self.x))

        check(isinstance(self.y, (list, np.ndarray)),
              'type(y) = %s must be of type list or np.array' % type(self.y))
        self.y = np.array(self.y)

        check(len(self.x) == len(self.y),
              'length of inputs (X, y) must be equal')
        self.num_measurements = len(self.x)

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that gpmap_type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              f'gpmap_type = {self.gpmap_type};'
              'must be "additive", "neighbor", or "pairwise"')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # clarify that X and y are the training datasets
        # (including validation sets)
        self.x_train, self.y_train = self.x, self.y

        # record sequence length for convenience
        self.L = len(self.x_train[0])

        # Encode sequences as features
        self.input_seqs_ohe, self.feature_names = \
            x_to_features(x=self.x_train,
                          alphabet=self.alphabet,
                          model_type=self.gpmap_type)

        # Leave out constant feature
        self.input_seqs_ohe = self.input_seqs_ohe[:, 1:]

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train).reshape(
            np.shape(self.y_train)[0], 1)


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
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        number_input_layer_nodes = len(self.input_seqs_ohe[0])+1
        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                output_shape=((len(self.input_seqs_ohe[0]),)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, len(
                self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + 1],
            output_shape=((1, )),
            trainable=False, name='Labels_input')(inputTensor)

        theta_regularizer = tf.keras.regularizers.l2(self.theta_regularization)
        phi = Dense(1,
                    name='phi',
                    kernel_regularizer=theta_regularizer)(sequence_input)

        # Make global epistasis layer
        self.phi_to_yhat_layer = \
            GlobalEpistasisLayer(K=ge_nonlinearity_hidden_nodes,
                                 eta_regularization=self.eta_regularization,
                                 monotonic=self.ge_nonlinearity_monotonic)
        yhat = self.phi_to_yhat_layer(phi)

        # Concatenate yhat and training labels
        yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
            [yhat, labels_input])

        # Compute likelihood
        likelihood_class = globals()[ge_noise_model_type + 'LikelihoodLayer']
        likelihood_object = likelihood_class(self.ge_heteroskedasticity_order,
                                             self.eta_regularization)
        outputTensor = likelihood_object(yhat_y_concat)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model


    @handle_errors
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
        yhat: (float or array-like)
            The nonlinear GE function evaluated at phi.

        """

        # Create phi_to_yhat function
        phi_input = Input((1,))
        yhat_output = self.phi_to_yhat_layer(phi_input)
        self.phi_to_yhat_func = Model(inputs=phi_input,
                                      outputs=yhat_output)

        # Predict yhat values
        yhat = self.phi_to_yhat_func.predict(phi)

        return yhat


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
        allowed: ['dna','rna','protein', 'protein*'].

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

    @handle_errors
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
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
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

        # useful tuple to check if some value is a number

        check(isinstance(self.x, (list, np.ndarray, pd.DataFrame, pd.Series)),
              'type(X) = %s must be of type list or np.array' % type(self.x))

        # check(isinstance(self.x[0], str),
        #      'type(x_train) = %s must be of type str' % type(self.x))

        check(isinstance(self.y, (list, np.ndarray, pd.DataFrame, pd.Series)),
              'type(y) = %s must be of type list or np.array' % type(self.y))

        if self.ct_n is not None:
            check(isinstance(self.ct_n,
                             (list, np.ndarray, pd.DataFrame, pd.Series)),
                  'type(ct_n) = %s; must be None or arrray,' % type(self.ct_n))

        # check(len(self.x) == len(self.y),
        #      'length of inputs (X, y) must be equal')

        # check that model type valid
        check(self.gpmap_type in {'additive', 'neighbor', 'pairwise'},
              f'model_type = {self.gpmap_type}; '
              'must be "additive", "neighbor", or "pairwise"')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        self.y, self.x = vec_data_to_mat_data(x_n=x,
                                              y_n=y,
                                              ct_n=ct_n)

        # record sequence length for convenience
        self.L = len(self.x[0])

        # Record number of bins
        self.Y = self.y.shape[1]
        self.all_y = np.arange(self.Y).astype(int)

        self.x_train, self.y_train = self.x, self.y

        # Encode sequences as features
        self.input_seqs_ohe, self.feature_names = \
            x_to_features(x=self.x_train,
                          alphabet=self.alphabet,
                          model_type=self.gpmap_type)

        # Leave out constant feature
        self.input_seqs_ohe = self.input_seqs_ohe[:, 1:]

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train)


    def define_model(self,
                     na_hidden_nodes=10):

        """
        Defines the architecture of the noise agnostic regression model.
        using the tensorflow.keras functional API. If custom_architecture is
        not None, this is used instead as the model architecture.

        parameters
        ----------
        na_hidden_nodes: (int)
            Number of nodes to use in the hidden layer of the measurement
            network of the GE model architecture.

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently
            fit to data.


        """

        # useful tuple to check if some value is a number

        check(isinstance(na_hidden_nodes, numbers.Integral),
              'na_hidden_nodes must be a number.')

        check(na_hidden_nodes > 0,
              'na_hidden_nodes must be greater than 0.')

        number_input_layer_nodes = len(self.input_seqs_ohe[0])+self.y.shape[1]

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:len(self.input_seqs_ohe[0])],
                                output_shape=((len(self.input_seqs_ohe[0]),)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, len(
                self.input_seqs_ohe[0]):len(
                self.input_seqs_ohe[0]) + self.y.shape[1]],
            output_shape=((1, )),
            trainable=False,
            name='Labels_input')(inputTensor)

        phi = Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.theta_regularization),
                    use_bias=True, name='phi')(sequence_input)

        intermediateTensor = Dense(na_hidden_nodes, activation='sigmoid')(phi)
        yhat = Dense(np.shape(self.y_train[0])[0],
                     name='yhat',
                     activation='softmax')(intermediateTensor)

        concatenateLayer = Concatenate(
            name='yhat_and_y_to_ll')([yhat, labels_input])
        outputTensor = MPALikelihoodLayer(
            number_bins=np.shape(self.y_train[0])[0])(concatenateLayer)

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

        # TODO: Replace these with layer names
        phi_index = 3
        yhat_index = 5

        # Form model using functional API in a loop, starting from
        # phi input, and ending on network output
        for layer in self.model.layers[phi_index:yhat_index]:
            next_input = layer(next_input)

        # Form gauge fixed GE_nonlinearity model
        na_model = Model(inputs=na_model_input, outputs=next_input)

        # compute the value of the nonlinearity for a given phi

        p_of_dot_given_phi = na_model.predict([phi])

        return p_of_dot_given_phi
