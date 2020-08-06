from mavenn.src.error_handling import handle_errors, check
from mavenn.src.UI import GlobalEpistasisModel, NoiseAgnosticModel
from mavenn.src.utils import _center_matrix, fix_gauge_neighbor_model, fix_gauge_pairwise_model
from mavenn.src.utils import onehot_encode_array, \
    _generate_nbr_features_from_sequences, _generate_all_pair_features_from_sequences

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model as kerasFunctionalModel # to distinguish from class name
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.constraints import non_neg as nonneg
import tensorflow.keras.backend as K

import pandas as pd
import numpy as np


@handle_errors
class Model:

    """
    Mavenn's model class that lets the user choose either
    global epistasis regression or noise agnostic regression


    attributes
    ----------

    regression_type: (str)
        variable that choose type of regression, valid options
        include 'GE', 'NA'

    X: (array-like)
        Input pandas DataFrame containing sequences. X are
        DNA, RNA, or protein sequences to be regressed over

    y: (array-like)
        y represents counts in bins, or continuous measurement values
        corresponding to the sequences X

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    learning_rate: (float)
        Learning rate of the optimizer.

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation.

    monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for NA regression.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    custom_architecture: (tf.model)
        Specify a custom neural network architecture (including both the
        G-P map and the measurement process) to fit to data.

    num_nodes_hidden_measurement_layer: (int)
        Number of nodes to use in the hidden layer of the measurement network
        of the GE model architecture.

    ohe_single_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.


    """

    def __init__(self,
                 regression_type,
                 X,
                 y,
                 alphabet,
                 gpmap_type='additive',
                 monotonic=True,
                 learning_rate=0.005,
                 test_size=0.2,
                 custom_architecture=None,
                 num_nodes_hidden_measurement_layer=50,
                 ohe_single_batch_size=10000):

        # set class attributes
        self.regression_type = regression_type
        self.X, self.y = X, y
        self.gpmap_type = gpmap_type
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.monotonic = monotonic
        self.alphabet = alphabet
        self.custom_architecture = custom_architecture
        self.num_nodes_hidden_measurement_layer = num_nodes_hidden_measurement_layer
        self.ohe_single_batch_size = ohe_single_batch_size

        # represents GE or NA model object, depending which is chosen.
        # attribute value is set below
        self.model = None

        # check that regression_type is valid
        check(self.regression_type in {'NA', 'GE'},
              'regression_type = %s; must be "NA", or  "GE"' %
              self.gpmap_type)

        # choose model based on regression_type
        if regression_type == 'GE':
            self.model = GlobalEpistasisModel(X=self.X,
                                              y=self.y,
                                              gpmap_type=self.gpmap_type,
                                              test_size=self.test_size,
                                              monotonic=self.monotonic,
                                              alphabet=self.alphabet,
                                              custom_architecture=self.custom_architecture,
                                              ohe_single_batch_size=self.ohe_single_batch_size)

            # TODO: take out define_model and compile model out of if-else conditional.
            # (check if this can break by doing that)
            self.define_model = self.model.define_model(num_nodes_hidden_measurement_layer=
                                                        self.num_nodes_hidden_measurement_layer,
                                                        custom_architecture=self.custom_architecture)

        elif regression_type == 'NA':
            self.model = NoiseAgnosticModel(X=self.X,
                                            y=self.y,
                                            gpmap_type=self.gpmap_type,
                                            test_size=self.test_size,
                                            alphabet=self.alphabet,
                                            custom_architecture=self.custom_architecture,
                                            ohe_single_batch_size=self.ohe_single_batch_size)

            self.define_model = self.model.define_model(num_nodes_hidden_measurement_layer=
                                                        self.num_nodes_hidden_measurement_layer,
                                                        custom_architecture=self.custom_architecture)

        self.compile_model(lr=self.learning_rate)

    @handle_errors
    def gauge_fix_model(self):

        """
        Method that gauge fixes the entire model (gpmap+measurement).

        parameters
        ----------
        None

        returns
        -------
        None

        """

        # TODO disable this method if user uses custom architecture

        # Helper variables used for gauge fixing gpmap trait parameters theta below.
        sequence_length = len(self.model.x_train[0])

        # Non-gauge fixed theta
        theta = self.model.model.layers[1].get_weights()[0]

        # The following conditionals gauge fix the gpmap parameters depending of the value of gpmap
        if self.gpmap_type == 'additive':

            # form pandas dataframe from theta_df so it can be used with the function _center_matrix
            theta_df = pd.DataFrame(theta.reshape(sequence_length, len(self.model.characters)), columns=self.model.characters)

            # compute mean centered, thus gauge-fixed, additive latent model
            gf_theta, theta_bar = _center_matrix(theta_df)
            gf_theta = gf_theta.values

            # compute offset term which separates latent trait from gauge fixed latent trait
            a = -np.sum(theta_bar)

        elif self.gpmap_type == 'neighbor':

            # compute gauge-fixed, neighbor latent model, and offset term
            gf_theta, a = fix_gauge_neighbor_model(sequence_length, len(self.model.characters), theta)

        elif self.gpmap_type == 'pairwise':

            # compute gauge-fixed pairwise latent model and offset term
            gf_theta, a = fix_gauge_pairwise_model(sequence_length, len(self.model.characters), theta)

        # add bias of default (non gauge fixed) model into additive shift
        bias = self.model.model.layers[1].get_weights()[1]
        a += bias[0]

        # The following variable unfixed_gpmap is a tf.keras backend function
        # which computes the non-gauge fixed value of the hidden node phi for a given input
        # this is  used to compute diffeomorphic scaling factor.
        unfixed_gpmap = K.function([self.model.model.layers[0].input], [self.model.model.layers[1].output])

        # compute unfixed phi using the function unfixed_gpmap with training sequences.
        unfixed_phi = unfixed_gpmap([self.model.input_seqs_ohe])

        # Compute diffeomorphic scaling factor which is used to rescale the parameters theta
        diffeomorphic_scale = np.sqrt(np.var(unfixed_phi[0]))

        # rescale the gf_theta
        theta_gf_rescaled = gf_theta.ravel() / diffeomorphic_scale

        # Default neural network weights that are non gauge fixed.
        # This will be used for updating the weights of the measurement
        # network after the gauge fixed neural network is define below.
        temp_weights = [layer.get_weights() for layer in self.model.model.layers]

        number_input_layer_nodes = len(self.model.input_seqs_ohe[0])
        inputTensor = Input((number_input_layer_nodes,), name='Sequence')

        phiPrime = Dense(1, name='phiPrime')(inputTensor)
        phi = Dense(1, name='phi')(phiPrime)

        # implement monotonicity constraints if GE regression
        if self.regression_type=='GE':
            if self.monotonic:
                intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid',
                                           kernel_constraint=nonneg())(phi)
                outputTensor = Dense(1, kernel_constraint=nonneg())(intermediateTensor)
            else:
                intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
                outputTensor = Dense(1)(intermediateTensor)
        elif self.regression_type=='NA':
            intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
            outputTensor = Dense(np.shape(self.model.y_train[0])[0], activation='softmax')(intermediateTensor)

        # create the gauge-fixed model:
        gf_model = kerasFunctionalModel(inputTensor, outputTensor)

        # set gpmap weights
        gf_model.layers[1].set_weights([theta_gf_rescaled.reshape(len(theta_gf_rescaled), 1), np.array([0])])

        # weight connecting phi to phiPrime with weight unity
        gf_model.layers[2].set_weights([np.array([diffeomorphic_scale]).reshape(1, 1), np.array([a])])

        # phiPrime to ge_model hidden
        gf_model.layers[3].set_weights(temp_weights[2])
        gf_model.layers[4].set_weights(temp_weights[3])

        # Update default neural network model with gauge-fixed model
        self.model.model = gf_model

        # The theta_gf attribute now contains gauge fixed parameters, and
        # can be obtained in raw form by accessing this attribute or can be
        # obtained a readable format by using the method return_theta
        self.model.theta_gf = theta_gf_rescaled.reshape(len(theta_gf_rescaled), 1)

    @handle_errors
    def fit(self,
            validation_split=0.2,
            epochs=50,
            verbose=1,
            early_stopping=True,
            early_stopping_patience=20):

        """

        Infers parameters, from data, for both the G-P map and the measurement process.

        parameters
        ----------
        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        epochs: (int>0)
            Maximum number of epochs to complete during training.

        verbose: (0 or 1, or boolean)
            Will show training progress if 1 or True, nothing if 0 or False.

        early_stopping: (bool)
            specifies whether to use early stopping or not

        early_stopping_patience: (int)
            If using early stopping, specifies the number of epochs to wait
            after a new optimum is identified.

        returns
        -------
        model: (tf.model)
            Trained neural network that can now be evaluated on the test set
            and used for predictions.

        """

        if early_stopping:
            callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  mode='auto',
                                                                  patience=early_stopping_patience)]
        else:
            callbacks = []

        history = self.model.model.fit(self.model.input_seqs_ohe,
                                       self.model.y_train,
                                       validation_split=validation_split,
                                       epochs=epochs,
                                       verbose=verbose,
                                       callbacks=callbacks
                                       )

        # gauge fix model after fitting
        #self.model.gauge_fix_model()
        #self.gauge_fix_model()
        self.gauge_fix_model()

        # update history attribute
        self.model.history = history
        return history

    @handle_errors
    def ge_nonlinearity(self,
                        phi):

        """
        Evaluate the GE nonlinearity at specified values of phi (the latent phenotype).

        parameters
        ----------

        phi: (array-like)
            Latent phenotype values at which to evaluate the GE nonlinearity

        returns
        -------
        y_hat: (array-like)
            GE nonlinearity evaluated on phi values

        """

        check(self.regression_type == 'GE', 'regression type must be "GE" for this function ')

        y_hat = self.model.ge_nonlinearity(phi)

        return y_hat

    @handle_errors
    def get_gpmap_parameters(self):

        """
        Returns gauge-fixed parameters for the G-P map as a pandas dataframe.
        The returned dataframe has two columns: name and values.
        The format of name is of the form:

        constant term: theta_0
        additive model term: theta_1:A
        neighbor model term: theta_1:AG
        pairwise model term: theta_1:A,5:G
        etc.
        the values column contain the parameter values corresponding to names.

        returns
        -------
        theta_gf: (pd DataFrame)
            Gauge-fixed G-P map parameters.
        """

        # temp variable to store characters.
        chars = self.model.characters

        # position and character indices
        char_indices = list(range(len(chars)))
        pos_indices = list(range(len(self.model.x_train[0])))

        # list that will contain parameter names
        names = []

        # list that will contain parameter values corresponding to names
        values = []

        # These parameters are gauge fixed are the model has been fit.
        if self.gpmap_type == 'additive':
            reshaped_theta = self.model.theta_gf.reshape(len(self.model.x_train[0]), len(chars))
            for position in pos_indices:
                for char in char_indices:
                    names.append('theta_' + str(position) + ':' + chars[char])
                    values.append(reshaped_theta[position][char])

        elif self.gpmap_type == 'neighbor':

            reshaped_theta = self.model.theta_gf.reshape(len(self.model.x_train[0]) - 1, len(chars), len(chars))

            # get parameters in tidy format
            for pos1 in pos_indices[:-1]:
                for char1 in char_indices:
                    for char2 in char_indices:
                        value = reshaped_theta[pos1][char1][char2]
                        name = f'theta_{pos1}:{chars[char1]}{chars[char2]}'
                        names.append(name)
                        values.append(value)

        elif self.gpmap_type == 'pairwise':

            # define helper variables
            sequenceLength = len(self.model.x_train[0])
            num_possible_pairs = int((sequenceLength * (sequenceLength - 1)) / 2)

            # reshape to num_possible_pairs by len(chars) by len(chars) array
            reshaped_theta = self.model.theta_gf.reshape(num_possible_pairs, len(chars), len(chars))

            pos_pair_num = 0
            for pos1 in pos_indices:
                for pos2 in pos_indices[pos1+1:]:
                    for char1 in char_indices:
                        for char2 in char_indices:
                            value = reshaped_theta[pos_pair_num][char1][char2]
                            name = f'theta_{pos1}:{chars[char1]},{pos2}:{chars[char2]}'
                            names.append(name)
                            values.append(value)
                    pos_pair_num += 1

        theta_tidy = pd.DataFrame(
            {'name': names,
             'value': values
             })

        return theta_tidy

    @handle_errors
    def na_noisemodel(self,
                      phi):

        """
        Evaluate the NA noise model at specified values of phi (the latent phenotype).

        parameters
        ----------

        phi: (array-like)
            Latent phenotype values at which to evaluate the noise model.

        returns
        -------
        pi: (array-like)
            Noise model evaluated on phi values.


        """

        check(self.regression_type == 'NA', 'regression type must be "GE" for this function ')

        pi = self.model.noise_model(phi)

        return pi

    @handle_errors
    def get_nn(self):

        """
        Returns the tf neural network used to represent the inferred model.
        """

        return self.model.model

    @handle_errors
    def compile_model(self,
                      optimizer=Adam,
                      lr=0.005):
        """
        This method will compile the model created in the constructor. The loss used will be
        log_poisson_loss for NA regression, or mean_squared_error for GE regression

        parameters
        ----------

        optimizer: (str)
            Specifies which optimizers to use during training. See
            'https://www.tensorflow.org/api_docs/python/tf/keras/optimizers',
            for a all available optimizers


        lr: (float)
            Learning rate of the optimizer.

        returns
        -------
        None

        """

        if self.regression_type == 'GE':
            self.model.model.compile(loss='mean_squared_error',
                                     optimizer=optimizer(lr=lr))

        elif self.regression_type == 'NA':
            self.model.model.compile(loss=tf.nn.log_poisson_loss,
                                     optimizer=optimizer(lr=lr))

    def gpmap(self,
              sequence):
        """

        Evaluates the latent phenotype phi on input sequences.

        parameters
        ----------
        sequence: (array-like of str)
            Sequence inputs representing DNA, RNA, or protein (whichever
            type of sequence the model was trained on). Input can must be
            an array of str, all the proper length.

        returns
        -------
        phi: (array-like of float)
            Array of latent phenotype values.

        """
        if self.gpmap_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            seqs_ohe = onehot_encode_array(sequence, self.model.characters)

        elif self.gpmap_type == 'neighbor':
            # one-hot encode sequences in batches in a vectorized way
            seqs_ohe = _generate_nbr_features_from_sequences(sequence, self.alphabet)

        elif self.gpmap_type == 'pairwise':
            # one-hot encode sequences in batches in a vectorized way
            seqs_ohe = _generate_all_pair_features_from_sequences(sequence, self.alphabet)

        # Form tf.keras function that will evaluate the value of gauge fixed latent phenotype
        gpmap_function = K.function([self.model.model.layers[0].input], [self.model.model.layers[1].output])

        # Compute latent phenotype values
        phi = gpmap_function([seqs_ohe])

        # Return latent phenotype values
        return phi

    def predict(self,
                data):

        """
        Make predictions for arbitrary input sequences. Note that this returns the output of
        the measurement process, not the latent phenotype.

        parameters
        ----------

        data: (array-like)
            Sequence data on which to make predictions.

        returns
        -------

        predictions: (array-like)
            An array of predictions. Note that this array will be 1D for GE regression,
            2D for NA regression.
        """

        # TODO need to do data validation here
        # e.g. check if data is already one-hot encoded

        if self.gpmap_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = onehot_encode_array(data, self.model.characters)

        elif self.gpmap_type == 'neighbor':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = _generate_nbr_features_from_sequences(data, self.alphabet)

        elif self.gpmap_type == 'pairwise':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = _generate_all_pair_features_from_sequences(data, self.alphabet)

        return self.model.model.predict(test_input_seqs_ohe)
