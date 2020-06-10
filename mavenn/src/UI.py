from mavenn.src.validate import validate_input
from mavenn.src.error_handling import handle_errors
from mavenn.src.utils import onehot_encode_array

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
3  Could add utils function that converts floating targets to counts in bins
   via rank ordering, thus allow real valued phenotype targets to be fit
   to NAR models.
4. Could add preprocessing step that scales the data before fitting and
   rescales the data after fitting to original scale.

"""


@handle_errors
class GlobalEpistasisModel:    

    """
    Class that implements global epistasis regression.


    attributes
    ----------

    df: (pd.DataFrame)
        Input pandas DataFrame containing design matrix X and targets Y. X are
        DNA, RNA, or protein sequences to be regressed over and Y are their
        corresponding targets/phenotype values.

    model_type: (str)
        Model type specifies the type of GE model the user wants to run.
        Three possible choices allowed: ['additive','neighbor','all-pairs']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein']

    sub_network_layers_nodes_dict: (dict)
        Dictionary that specifies the number of layers and nodes in each subnetwork layer

    """

    def __init__(self,
                 df,
                 model_type,
                 test_size=0.2,
                 alphabet_dict='dna',
                 sub_network_layers_nodes_dict=None):

        # set class attributes
        self.df = df
        self.model_type = model_type
        self.test_size = test_size
        self.alphabet_dict = alphabet_dict
        self.sub_network_layers_nodes_dict = sub_network_layers_nodes_dict

        # class attributes that are not parameters
        self.history = None
        self.model = None

        # perform input checks to validate attributes
        self._input_checks()

        self.x_train, self.y_train = self.df['sequence'].values, self.df['values'].values

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
        elif self.self.alphabet_dict == 'protein':

            # this should be called amino-acids
            # need to figure out way to deal with
            # naming without changing a bunch of
            # unnecessary refactoring.
            self.bases = ['A', 'C', 'D', 'E', 'F',
                          'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R',
                          'S', 'T', 'V', 'W', 'Y']

        # one-hot encode sequences in batches in a vectorized way
        self.input_seqs_ohe = onehot_encode_array(self.x_train, self.bases)

        # check if this is strictly required by tf
        self.y_train = np.array(self.y_train).reshape(self.y_train.shape[0], 1)

    def _input_checks(self):

        """
        Validate parameters passed to the GlobalEpistasis constructor
        """
        # validate input df
        self.df = validate_input(self.df)

    def _generate_nbr_features_from_sequences(self,
                                              sequences):

        """
        Method that takes in sequences are generates sequences
        with neighbor features

        parameters
        ----------

        sequences: (array-like)
            array contains raw input sequences

        returns
        -------
        nbr_sequences: (array-like)
            Data Frame of sequences where each row contains a sequence example
            with neighbor features

        """

        pass

    def _generate_all_pair_features_from_sequences(self,
                                                   sequences):

        """
        Method that takes in sequences are generates sequences
        with all pair features

        parameters
        ----------

        sequences: (array-like)
            array contains raw input sequences

        returns
        -------

        all_pairs_sequences: (array-like)
            Data Frame of sequences where each row contains a sequence example
            with all-pair features

        """

        pass

    # JBK: User should be able to set these parameters in the constructor
    def define_model(self,
                     monotonic=True,
                     regularization=None,
                     activations='sigmoid'):

        """
        Method that will define the architecture of the global epistasis model.
        using the tensorflow.keras functional api. If the subnetwork architecture
        is not None, than archicture is constructed via the dict sub_network_layers_nodes_dict

        parameters
        ----------

        monotonic: (boolean)
            If True, than weights in subnetwork will be constrained to be non-negative.
            Doing so will constrain the global epistasis non-linearity to be monotonic

        regularization: (str)
            String that specifies type of regularization to use. Valid choices include
            ['l2','dropout', ... link to tf docs].

        activations: (str)
            Activation function used in the non-linear sub-network. Link to allowed
            activation functions from TF docs...

        returns
        -------

        model: (tf.model)
            A tensorflow model that can be compiled and subsequently fit to data.


        """

        number_input_layer_nodes = len(self.input_seqs_ohe[0])
        inputTensor = Input((number_input_layer_nodes,), name='Sequence')

        phi = Dense(1, use_bias=True)(inputTensor)

        intermediateTensor = Dense(50, activation='sigmoid', kernel_constraint=nonneg())(phi)
        outputTensor = Dense(1, kernel_constraint=nonneg())(intermediateTensor)

        # create the model:

        model = Model(inputTensor, outputTensor)
        self.model = model
        return model

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
        model: (tf.model)
            A compiled tensorflow.keras model that can be fit to data.


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

        test_input_seqs_ohe = onehot_encode_array(data, self.bases)
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
                    input_range):

        """
        Method used to plot GE non-linearity.

        parameters
        ----------

        input_range: (array-like)
            data which will be input to the GE nonlinearity


        returns
        -------
        ge_nonlinearity: (array-like function)
            the nonlinear GE function.

        """

        # TODO input checks on input_range

        ge_model_input = Input((1,))
        next_input = ge_model_input

        # TODO need to fix this hardcoded 2 depending on model architecture
        for layer in self.model.layers[-2:]:
            next_input = layer(next_input)

        ge_model = Model(inputs=ge_model_input, outputs=next_input)

        ge_nonlinearity = ge_model.predict(input_range)

        return ge_nonlinearity


class NoiseAgnosticModel:

    """
    Class that implements Noise agnostic regression.


    attributes
    ----------

    df: (pd.DataFrame)
        Input pandas DataFrame containing design matrix X and targets Y. X are
        DNA, RNA, or protein sequences to be regressed over and Y are their
        corresponding counts in bin values.

    model_type: (str)
        Model type specifies the type of NAR model the user wants to run.
        Three possible choices allowed: ['additive','neighbor','all-pairs']

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    alphabet_dict: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein']

    noise_model_layers_nodes_dict: (dict)
        Dictionary that specifies the number of layers and nodes in each noise model layer

    """

    def __init__(self,
                 df,
                 model_type,
                 test_size=0.2,
                 alphabet_dict='dna',
                 noise_model_layers_nodes_dict=None):

        # set class attributes
        self.df = df
        self.model_type = model_type
        self.test_size = test_size
        self.alphabet_dict = alphabet_dict
        self.noise_model_layers_nodes_dict = noise_model_layers_nodes_dict

        # perform input checks to validate attributes
        self._input_checks()

        pass

    def _input_checks(self):

        """
        Validate parameters passed to the NoiseAgnosticRegression constructor
        """
        pass

    def define_model(self,
                     nonneg_weights=True,
                     regularization=None,
                     activations='sigmoid'):

        """
        Method that will define the architecture of the global epistasis model.
        using the tensorflow.keras functional api. If the subnetwork architecture
        is not None, than archicture is constructed via the dict sub_network_layers_nodes_dict

        parameters
        ----------

        nonneg_weights: (boolean)
            If True, than weights in output will be constrained to be non-negative.

        regularization: (str)
            String that specifies type of regularization to use. Valid choices include
            ['l2','dropout', ... link to tf docs].

        activations: (str)
            Activation function used in the non-linear sub-network. Link to allowed
            activation functions from TF docs...


        returns
        -------

        model: (tf.model)
            A tensorflow.keras model that can be compiled and subsequently fit to data.


        """

        pass

    def compile_model(self,
                      model,
                      optimizer='Adam',
                      lr=0.0001):

        """
        This method will compile the model created in the define_model method.
        The loss used will be log_poisson_loss.

        parameters
        ----------

        model: (tf.model)
            A tensorflow.keras model to be compiled

        optimizer: (str)
            Specifies which optimizers to use during training. Valid choices include
            ['Adam', 'SGD', 'RMSPROP', ... link to keras documentation']

        lr: (float)
            Learning rate of the optimizer.

        returns
        -------
        model: (tf.model)
            A compiled tensorflow.keras model that can be fit to data.


        """
        pass

    def model_fit(self,
                  model,
                  sequences,
                  cts_in_bins,
                  validation_split=0.2,
                  epochs=100,
                  verbose=1):

        """

        Method that will fit the global epistasis model to data.

        parameters
        ----------

        model: (tf.model)
            A compiled tensorflow model ready to be fit to data.

        sequences: (array-like)
            Array of sequences that will be used in training.

        cts_in_bins: (array-like)
            Array of counts in bins corresponding to sequences that
            will be used as targets during training.

        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        epochs: (int>0)
            Number of epochs to complete during training.

        verbose: (0 or 1, or boolean)
            Boolean variable that will show training progress if 1 or True, nothing
            if 0 or False.

        returns
        -------
        model: (tf.model)
            Trained model that can now be evaluated on the test set
            and used for predictions.

        """

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


