"""
The following are imports that will go in the main __init__ file of the package.
The interface definitions for imports are below

from mpathic.src import GlobalEpistasis
from mpathic.src import NoiseAgnosticRegression


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

# TODO: implement error handling to decorate interface methods.


class GlobalEpistasis:

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

        # perform input checks to validate attributes
        self._input_checks()

        pass

    def _input_checks(self):

        """
        Validate parameters passed to the GlobalEpistasis constructor
        """
        pass

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

        pass

    def compile_model(self,
                      model,
                      optimizer='Adam',
                      lr=0.0001,
                      metrics=None):

        """
        This method will compile the model created in the define_model method.
        Loss is mean squared error.

        parameters
        ----------

        model: (tf.model)
            A tensorflow.keras model to be compiled

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
        pass

    def model_fit(self,
                  model,
                  sequences,
                  phenotypes,
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

        phenotypes: (array-like)
            Array of targets corresponding to sequences that will
            be used during training.

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

    def model_predict(self,
                       model,
                       data):

        """
        Method to make predictions from trained model

        parameters
        ----------

        model: (tf.model)
            A trained tensorflow model.

        data: (array-like)
            Data on which to make predictions.

        returns
        -------

        predictions: (array-like)
            An array of predictions
        """

        pass

    def plot_losses(self,
                    trained_model):

        """
        Method used to display loss values.

        parameters
        ----------
        trained_model: (tf.model)
            trained_model from which loss values vs. epochs can be plotted

        data: (array-like)
            Data on which to make predictions.

        returns
        -------
        None

        """

        pass

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


class NoiseAgnosticRegression:

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
