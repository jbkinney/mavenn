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
    # JBK: Looks good, but should there be more optional attributes, e.g. to limit training time, specify # epochs, etc?  
    # Also, monotonicity, activations, regularization?

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

        # perform input checks to validate attributes
        self._input_checks()



        #
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df['sequence'].values,
        #                                                                         self.df['values'].values)

        self.x_train, self.y_train = self.df['sequence'].values, self.df['values'].values

        print('One-hot encoding...')
        # # onehot-encode sequences
        # self.input_seqs_ohe = []
        # for _ in range(len(self.x_train)):
        #     self.input_seqs_ohe.append(onehot_sequence(self.x_train[_]).ravel())
        #
        # # turn lists into np arrays for consumption by tf
        # self.input_seqs_ohe = np.array(self.input_seqs_ohe)


        # TODO: move the one-hot encoding to utils
        sequence_length = len(self.x_train[0])

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

        # # one-hot encode sequences in batches in a vectorized way
        # ohe_single_batch_size = 10000
        # # container list for batches of oh-encoded sequences
        # input_seqs_ohe_batches = []
        #
        # # partitions of batches
        # ohe_batches = np.arange(0, len(self.x_train), ohe_single_batch_size)
        # for ohe_batch_index in range(len(ohe_batches)):
        #     if ohe_batch_index == len(ohe_batches) - 1:
        #         # OHE remaining sequences (that are smaller than batch size)
        #         input_seqs_ohe_batches.append(
        #             onehot_sequence(''.join(self.x_train[ohe_batches[ohe_batch_index]:]))
        #             .reshape(-1, sequence_length, len(self.bases)))
        #     else:
        #         # OHE sequences in batches
        #         input_seqs_ohe_batches.append(onehot_sequence(
        #             ''.join(self.x_train[ohe_batches[ohe_batch_index]:ohe_batches[ohe_batch_index + 1]]))
        #               .reshape(-1, sequence_length, len(self.bases)))
        #
        # # this array will contain the one-hot encoded sequences
        # self.input_seqs_ohe = np.array([])
        #
        # # concatenate all the oh-encoded batches
        # for batch_index in range(len(input_seqs_ohe_batches)):
        #     self.input_seqs_ohe = np.concatenate([self.input_seqs_ohe, input_seqs_ohe_batches[batch_index]
        #                                          .ravel()]).copy()
        #
        # # reshape so that shape of oh-encoded array is [number samples, sequence_length*alphabet_dict]
        # self.input_seqs_ohe = self.input_seqs_ohe.reshape(len(self.x_train), sequence_length * len(self.bases)).copy()

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
            A tensorflow.keras model that can be compiled and subsequently fit to data.


        """

        number_input_layer_nodes = len(self.input_seqs_ohe[0])
        inputTensor = Input((number_input_layer_nodes,), name='Sequence')

        phi = Dense(1, use_bias=True)(inputTensor)

        intermediateTensor = Dense(50, activation='sigmoid', kernel_constraint=nonneg())(phi)
        # intermediateTensor = Dense(20,activation='sigmoid',kernel_constraint=nonneg())(intermediateTensor)
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
                  verbose=1):

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

        returns
        -------
        model: (tf.model)
            Trained model that can now be evaluated on the test set
            and used for predictions.

        """

        history = self.model.fit(self.input_seqs_ohe,
                                 self.y_train,
                                 validation_split=validation_split,
                                 epochs=epochs,
                                 verbose=1)

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
        # check if data is already one-hot encoded

        # test_input_seqs_ohe = []
        # for _ in range(len(data)):
        #     test_input_seqs_ohe.append(onehot_sequence(data[_]).ravel())
        #
        # # turn lists into np arrays for consumption by tf
        # test_input_seqs_ohe = np.array(test_input_seqs_ohe)

        test_input_seqs_ohe = onehot_encode_array(data, self.bases)
        return self.model.predict(test_input_seqs_ohe)

    # JBK: we should discuss the suite of diagnostics we want to provide. 
    def plot_losses(self):

        """
        Method used to display loss values.

        returns
        -------
        None

        """

        plt.figure()
        plt.plot(self.history.history['loss'], color='blue')
        plt.plot(self.history.history['val_loss'], color='orange')
        plt.title('Model loss', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.legend(['train', 'validation'])
        plt.show()
      
    # JBK: we should discuss the suite of diagnostics we want to provide. 
    # Also, the user might want a simple function object that represents the underlying nonlinearity,
    # One they can evaluate on any inptut they provide. 
    def plot_GE_nonlinearity(self,
                    trained_model,
                    data):

        """
        Method used to plot GE non-linearity.

        parameters
        ----------
        trained_model: (tf.model)
            trained_model from which loss values vs. epochs can be plotted

        data: (array-like)
            Data which will be used to make latent model predictions, which
            will then used to render the plot.

        returns
        -------
        None

        """

        pass

# JBK: NoiseAgnosticModel instead?
# Similar comments apply as for GlobalEpistais.
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


