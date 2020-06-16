from mavenn.src.validate import validate_input
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.utils import onehot_encode_array, \
    _generate_nbr_features_from_sequences, _generate_all_pair_features_from_sequences

import numpy as np
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Lambda, Concatenate
from tensorflow.keras.constraints import non_neg as nonneg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

"""
NOTE: could put methods that are common to both classes in
in a utils.py module, e.g. _generate_all_pair_features_from_sequences()

Optional tasks
1. Could add a method to display additive models as sequences logos.
2. Could add mutual information approximator for trained NAR model.
"""


@handle_errors
class GlobalEpistasisModel:    

    """
    Class that implements global epistasis regression.


    attributes
    ----------

    X: (array-like)
        Input pandas DataFrame containing sequences. X are
        DNA, RNA, or protein sequences to be regressed over

    y: (array-like)
        y represents counts in bins corresponding to the sequences X

    model_type: (str)
        Model type specifies the type of GE model the user wants to run.
        Three possible choices allowed: ['additive','neighbor','all-pairs']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    custom_architecture: (tf.model)
        a custom neural network architecture that replaces the
        default architecture implemented.

    ohe_single_batch_size: (int)
        integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    """

    def __init__(self,
                 X,
                 y,
                 model_type='additive',
                 test_size=0.2,
                 alphabet_dict='dna',
                 custom_architecture=None,
                 ohe_single_batch_size=10000):

        # set class attributes
        self.X, self.y = X, y
        self.model_type = model_type
        self.test_size = test_size
        self.alphabet_dict = alphabet_dict
        self.custom_architecture = custom_architecture
        self.ohe_single_batch_size = ohe_single_batch_size

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # perform input checks to validate attributes
        self._input_checks()

        self.x_train, self.y_train = self.X, self.y

        print('One-hot encoding...')
        # # onehot-encode sequences
        # self.input_seqs_ohe = []
        # for _ in range(len(self.x_train)):
        #     self.input_seqs_ohe.append(onehot_sequence(self.x_train[_]).ravel())
        #
        # # turn lists into np arrays for consumption by tf
        # self.input_seqs_ohe = np.array(self.input_seqs_ohe)

        if self.alphabet_dict == 'dna':
            self.bases = ['A', 'C', 'G', 'T']
        elif self.alphabet_dict == 'rna':
            self.bases = ['A', 'C', 'G', 'U']
        elif self.alphabet_dict == 'protein':

            # this should be called amino-acids
            # need to figure out way to deal with
            # naming without changing a bunch of
            # unnecessary refactoring.
            self.bases = ['A', 'C', 'D', 'E', 'F',
                          'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R',
                          'S', 'T', 'V', 'W', 'Y']

        if model_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.bases, self.ohe_single_batch_size)

        elif model_type == 'neighbor':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = _generate_nbr_features_from_sequences(self.x_train, self.alphabet_dict)

        elif model_type == 'pairwise':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = _generate_all_pair_features_from_sequences(self.x_train, self.alphabet_dict)

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train).reshape(np.shape(self.y_train)[0], 1)

    def _input_checks(self):

        """
        Validate parameters passed to the GlobalEpistasis constructor
        """
        # validate input df
        #self.df = validate_input(self.df)

        check(isinstance(self.X, (list, np.ndarray)),
              'type(X) = %s must be of type list or np.array' % type(self.X))

        check(isinstance(self.y, (list, np.ndarray)),
              'type(y) = %s must be of type list or np.array' % type(self.y))

        # check that model type valid
        check(self.model_type in {'additive', 'neighbor', 'pairwise'},
              'model_type = %s; must be "additive", "neighbor", or "pairwise"' %
              self.model_type)

    # JBK: User should be able to set these parameters in the constructor
    def define_model(self,
                     monotonic=True,
                     custom_architecture=None):

        """
        Method that will define the architecture of the global epistasis model.
        using the tensorflow.keras functional api. If the subnetwork architecture
        is not None, than archicture is constructed via the self.custom_architecture
        keyword

        parameters
        ----------

        monotonic: (boolean)
            Indicates whether to use monotonicity constraint in GE nonlinear function.
            If true then weights of GE nonlinear function will be constraned to
            be non-negative.

        custom_architecture: (tf.model)
            a custom neural network architecture that replaces the
            default architecture implemented.

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        # user has not provided custom architecture, implement a default architecture
        if custom_architecture is None:

            number_input_layer_nodes = len(self.input_seqs_ohe[0])
            inputTensor = Input((number_input_layer_nodes,), name='Sequence')

            phi = Dense(1, use_bias=True)(inputTensor)

            # implement monotonicity constraints
            if monotonic:
                intermediateTensor = Dense(50, activation='sigmoid', kernel_constraint=nonneg())(phi)
                outputTensor = Dense(1, kernel_constraint=nonneg())(intermediateTensor)
            else:
                intermediateTensor = Dense(50, activation='sigmoid')(phi)
                outputTensor = Dense(1)(intermediateTensor)

            # create the model:

            model = Model(inputTensor, outputTensor)
            self.model = model

            return model

        # if user has provided custom architecture
        else:
            self.model = custom_architecture
            return custom_architecture

    def return_model(self):
        """

        returns (GE model)
            Helper method that returns the model attribute,
            so that it may easily accessible from the mavenn
            Model class
        """

        return self.model

    def compile_model(self,
                      optimizer='Adam',
                      lr=0.00005,
                      metrics=None):

        """
        This method will compile the model created in the define_model method.
        Loss is mean squared error.

        parameters
        ----------
        optimizer: (str)
            Specifies which optimizers to use during training. Valid choices include
            ['Adam', 'SGD', 'RMSPROP', ... link to keras documentation']

        lr: (float)
            Learning rate of the optimizer.

        metrics: (str)
            Optional metrics to track as MSE loss is minimized, e.g. mean absolute
            error. Link to allowed metrics from keras documention...


        returns
        -------
        None


        """
        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(lr=lr),
                           metrics=['mean_absolute_error'])

    def fit(self,
            validation_split=0.2,
            epochs=50,
            verbose=1,
            use_early_stopping=True,
            early_stopping_patience=50
           ):

        """

        Method that will fit the global epistasis model to data.

        parameters
        ----------
        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        epochs: (int>0)
            Number of epochs to complete during training.

        verbose: (0 or 1, or boolean)
            Boolean variable that will show training progress if 1 or True, nothing
            if 0 or False.

        use_early_stopping: (bool)
            specifies whether to use early stopping or not

        early_stopping_patience: (int)
            number of epochs to wait before executing early stopping.


        returns
        -------
        model: (tf.model)
            Trained model that can now be evaluated on the test set
            and used for predictions.

        """

        if use_early_stopping:
            esCallBack = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  mode='auto',
                                                                  patience=early_stopping_patience)

            history = self.model.fit(self.input_seqs_ohe,
                                     self.y_train,
                                     validation_split=validation_split,
                                     epochs=epochs,
                                     verbose=verbose,
                                     callbacks=[esCallBack]
                                     )

        else:
            history = self.model.fit(self.input_seqs_ohe,
                                     self.y_train,
                                     validation_split=validation_split,
                                     epochs=epochs,
                                     verbose=verbose,
                                     )

        self.history = history
        return history

    # JBK: perhaps just call this "evaluate". Also, why pass "model" argument?
    def model_evaluate(self,
                       model,
                       data):

        """
        Method to evaluate trained model.

        parameters
        ----------

        model: (tf.model)
            A trained tensorflow model.

        data: (array-like)
            Data to be evaluate model on.

        returns
        -------
        Value of loss on the test set.
        """

        pass

    def predict(self,
                data):

        """
        Method to make predictions from trained model

        parameters
        ----------

        data: (array-like)
            Data on which to make predictions.

        returns
        -------

        predictions: (array-like)
            An array of predictions
        """

        # TODO need to do data validation here
        # e.g. check if data is already one-hot encoded

        if self.model_type=='additive':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = onehot_encode_array(data, self.bases)

        elif self.model_type=='neighbor':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = _generate_nbr_features_from_sequences(data, self.alphabet_dict)

        elif self.model_type=='pairwise':
            # one-hot encode sequences in batches in a vectorized way
            test_input_seqs_ohe = _generate_all_pair_features_from_sequences(data, self.alphabet_dict)

        return self.model.predict(test_input_seqs_ohe)

    def return_loss(self):

        """
        Method that returns loss values.

        returns
        -------
        history: (object)
            self.attribute/object that contains loss history

        """

        return self.history

    def ge_nonlinearity(self,
                        sequences,
                        input_range=None,
                        gauge_fix=True):

        """
        Method used to plot GE non-linearity.

        parameters
        ----------

        sequences: (array-like)
            sequences for which the additive trait will be computed.

        input_range: (array-like)
            data range which will be input to the GE nonlinearity.
            If this is none than range will be determined from min
            and max of the latent trait

        gauge_fix: (bool)
            if true parameters used to compute latent trait will be
            gauge fixed

        returns
        -------
        ge_nonlinearity: (array-like function)
            the nonlinear GE function.

        """

        # TODO input checks on input_range

        # one hot encode sequences and then subsequently use them
        # to compute latent trait

        if self.model_type=='additive':
            # one-hot encode sequences in batches in a vectorized way
            sequences_ohe = onehot_encode_array(sequences, self.bases)

        elif self.model_type=='neighbor':
            # one-hot encode sequences in batches in a vectorized way
            sequences_ohe = _generate_nbr_features_from_sequences(sequences, self.alphabet_dict)

        elif self.model_type=='pairwise':
            # one-hot encode sequences in batches in a vectorized way
            sequences_ohe = _generate_all_pair_features_from_sequences(sequences, self.alphabet_dict)

        get_1st_layer_output = K.function([self.model.layers[0].input], [self.model.layers[1].output])
        latent_trait = get_1st_layer_output([sequences_ohe])

        # tf adds an extra dimensions, so we will remove it
        latent_trait = latent_trait[0].ravel().copy()

        ge_model_input = Input((1,))
        next_input = ge_model_input

        # TODO need to fix this hardcoded 2 depending on model architecture
        for layer in self.model.layers[-2:]:
            next_input = layer(next_input)

        ge_model = Model(inputs=ge_model_input, outputs=next_input)

        if input_range is None:
            input_range = np.linspace(min(latent_trait), max(latent_trait), 1000)
            ge_nonlinearity = ge_model.predict(input_range)

            # if input range not provided, return input range
            # so ge nonliearity can be plotted against it.
            return ge_nonlinearity, input_range, latent_trait
        else:
            ge_nonlinearity = ge_model.predict(input_range)
            return ge_nonlinearity


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

    model_type: (str)
        Model type specifies the type of NAR model the user wants to run.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein']

    custom_architecture: (tf.model)
        a custom neural network architecture that replaces the
        default architecture implemented.

    ohe_single_batch_size: (int)
        integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    """

    def __init__(self,
                 X,
                 y,
                 model_type='additive',
                 test_size=0.2,
                 alphabet_dict='dna',
                 custom_architecture=None,
                 ohe_single_batch_size=10000):

        # set class attributes
        self.X = X
        self.y = y
        self.model_type = model_type
        self.test_size = test_size
        self.alphabet_dict = alphabet_dict
        self.custom_architecture = custom_architecture
        self.ohe_single_batch_size = ohe_single_batch_size

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # perform input checks to validate attributes
        self._input_checks()

        self.x_train, self.y_train = self.X, self.y

        print('One-hot encoding...')
        # # onehot-encode sequences
        # self.input_seqs_ohe = []
        # for _ in range(len(self.x_train)):
        #     self.input_seqs_ohe.append(onehot_sequence(self.x_train[_]).ravel())
        #
        # # turn lists into np arrays for consumption by tf
        # self.input_seqs_ohe = np.array(self.input_seqs_ohe)

        if self.alphabet_dict == 'dna':
            self.bases = ['A', 'C', 'G', 'T']
        elif self.alphabet_dict == 'rna':
            self.bases = ['A', 'C', 'G', 'U']
        elif self.alphabet_dict == 'protein':

            # this should be called amino-acids
            # need to figure out way to deal with
            # naming without changing a bunch of
            # unnecessary refactoring.
            self.bases = ['A', 'C', 'D', 'E', 'F',
                          'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R',
                          'S', 'T', 'V', 'W', 'Y']

        if model_type=='additive':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = onehot_encode_array(self.x_train, self.bases, self.ohe_single_batch_size)

        elif model_type=='neighbor':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = _generate_nbr_features_from_sequences(self.x_train, self.alphabet_dict)

        elif model_type=='pairwise':
            # one-hot encode sequences in batches in a vectorized way
            self.input_seqs_ohe = _generate_all_pair_features_from_sequences(self.x_train, self.alphabet_dict)

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train)

    def _input_checks(self):

        """
        Validate parameters passed to the NoiseAgnosticRegression constructor
        """
        pass

    def define_model(self,
                     monotonic=True,
                     custom_architecture=None):

        """
        Method that will define the architecture of the global epistasis model.
        using the tensorflow.keras functional api. If the subnetwork architecture
        is not None, than archicture is constructed via the dict sub_network_layers_nodes_dict

        parameters
        ----------

        monotonic: (boolean)
            If True, than weights in noise model will be constrained to be non-negative.

        custom_architecture: (tf.model)
            a custom neural network architecture that replaces the
            default architecture implemented.

        returns
        -------

        model: (tf.model)
            A tensorflow.keras model that can be compiled and subsequently fit to data.


        """

        if custom_architecture is None:
            number_input_layer_nodes = len(self.input_seqs_ohe[0])
            inputTensor = Input((number_input_layer_nodes,), name='Sequence')

            phi = Dense(1, use_bias=True, name='additive_weights')(inputTensor)

            # implement monotonicity constraints
            if monotonic:
                intermediateTensor = Dense(10, activation='sigmoid', kernel_constraint=nonneg())(phi)
                outputTensor = Dense(np.shape(self.y_train[0])[0], activation='softmax', kernel_constraint=nonneg())(
                    intermediateTensor)
            else:
                intermediateTensor = Dense(10, activation='sigmoid')(phi)
                outputTensor = Dense(np.shape(self.y_train[0])[0], activation='softmax')(intermediateTensor)


            # #create the model:
            model = Model(inputTensor, outputTensor)
            self.model = model
            return model
        else:
            self.model = custom_architecture
            return custom_architecture

    def compile_model(self,
                      lr=0.005):

        """
        This method will compile the model created in the define_model method.
        The loss used will be log_poisson_loss.

        parameters
        ----------

        optimizer: (str)
            Specifies which optimizers to use during training. Valid choices include
            ['Adam', 'SGD', 'RMSPROP', ... link to keras documentation']

        lr: (float)
            Learning rate of the optimizer.

        metrics: (str)
            Optional metrics to track as MSE loss is minimized, e.g. mean absolute
            error. Link to allowed metrics from keras documention...


        returns
        -------
        None

        """

        self.model.compile(loss=tf.nn.log_poisson_loss,
                           optimizer=Adam(lr=lr),
                           metrics=['categorical_accuracy'])

    def fit(self,
            validation_split=0.2,
            epochs=50,
            verbose=1,
            use_early_stopping=True,
            early_stopping_patience=50):

        """

        Method that will fit the noise agnostic model to data.

        parameters
        ----------
        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        epochs: (int>0)
            Number of epochs to complete during training.

        verbose: (0 or 1, or boolean)
            Boolean variable that will show training progress if 1 or True, nothing
            if 0 or False.

        use_early_stopping: (bool)
            specifies whether to use early stopping or not

        early_stopping_patience: (int)
            number of epochs to wait before executing early stopping.


        returns
        -------
        model: (tf.model)
            Trained model that can now be evaluated on the test set
            and used for predictions.

        """

        if use_early_stopping:
            esCallBack = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  mode='auto',
                                                                  patience=early_stopping_patience)

            history = self.model.fit(self.input_seqs_ohe,
                                     self.y_train,
                                     validation_split=validation_split,
                                     epochs=epochs,
                                     verbose=verbose,
                                     callbacks=[esCallBack]
                                     )

        else:
            history = self.model.fit(self.input_seqs_ohe,
                                     self.y_train,
                                     validation_split=validation_split,
                                     epochs=epochs,
                                     verbose=verbose,
                                     )

        self.history = history
        return history

    def model_evaluate(self,
                       model,
                       data):

        """
        Method to evaluate trained model.

        parameters
        ----------

        model: (tf.model)
            A trained tensorflow model.

        data: (array-like)
            Data to be evaluate model on.

        returns
        -------
        Value of loss on the test set.
        """

        pass

    def return_loss(self):

        """
        Method that returns loss values.

        returns
        -------
        history: (object)
            self.attribute/object that contains loss history

        """

        return self.history

    def return_model(self):
        """

        returns (GE model)
            Helper method that returns the model attribute,
            so that it may easily accessible from the mavenn
            Model class
        """

        return self.model

    def noise_model(self,
                    sequences=None,
                    input_range=None,
                    gauge_fix=True):

        """
        Method used to compute NA noise model.

        parameters
        ----------

        sequences: (array-like)
            sequences for which the additive trait will be computed.

        input_range: (array-like)
            data range which will be input to the noise model.
            If this is none than range will be determined from min
            and max of the latent trait

        gauge_fix: (bool)
            if true parameters used to compute latent trait will be
            gauge fixed

        returns
        -------
        noise_model: (array-like function)
            the noise model mapping latent trait to bins.

        """

        # TODO input checks on input_range and sequences

        # one hot encode sequences and then subsequently use them
        # to compute latent trait

        # compute latent trait if sequences given
        if sequences is not None:

            if self.model_type=='additive':
                # one-hot encode sequences in batches in a vectorized way
                sequences_ohe = onehot_encode_array(sequences, self.bases)

            elif self.model_type=='neighbor':
                # one-hot encode sequences in batches in a vectorized way
                sequences_ohe = _generate_nbr_features_from_sequences(sequences, self.alphabet_dict)

            elif self.model_type=='pairwise':
                # one-hot encode sequences in batches in a vectorized way
                sequences_ohe = _generate_all_pair_features_from_sequences(sequences, self.alphabet_dict)

            get_1st_layer_output = K.function([self.model.layers[0].input], [self.model.layers[1].output])
            latent_trait = get_1st_layer_output([sequences_ohe])

            # tf adds an extra dimensions, so we will remove it
            latent_trait = latent_trait[0].ravel().copy()

        noise_model_input = Input((1,))
        next_input = noise_model_input

        # TODO need to fix this hardcoded 2 depending on model architecture
        for layer in self.model.layers[-2:]:
            next_input = layer(next_input)

        noise_model = Model(inputs=noise_model_input, outputs=next_input)

        if input_range is None and sequences is not None:
            input_range = np.linspace(min(latent_trait), max(latent_trait), 1000)
            noise_function = noise_model.predict(input_range)

            # if input range not provided, return input range
            # so ge nonliearity can be plotted against it.
            return noise_function, input_range, latent_trait
        else:
            noise_function = noise_model.predict(input_range)
            return noise_function
