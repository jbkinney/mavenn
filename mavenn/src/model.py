from mavenn.src.error_handling import handle_errors, check
from mavenn.src.UI import GlobalEpistasisModel, NoiseAgnosticModel


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
        y represents counts in bins corresponding to the sequences X

    model_type: (str)
        Model type specifies the type of GE model the user wants to run.
        Three possible choices allowed: ['additive','neighbor','all-pairs']

    learning_rate: (float)
            Learning rate of the optimizer.

    test_size: (float in (0,1))
        Fraction of data to be set aside as unseen test data for model evaluation
        error.

    monotonic: (boolean)
        Indicates whether to use monotonicity constraint in GE nonlinear function
        or NA noise model. If true then weights of GE nonlinear function, or NA
        noise model, will be constraned to be non-negative.

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
                 regression_type,
                 X,
                 y,
                 model_type,
                 learning_rate=0.005,
                 test_size=0.2,
                 monotonic=True,
                 alphabet_dict='dna',
                 custom_architecture=None,
                 ohe_single_batch_size=10000):

        # set class attributes
        self.regression_type = regression_type
        self.X, self.y = X, y
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.monotonic = monotonic
        self.alphabet_dict = alphabet_dict
        self.custom_architecture = custom_architecture
        self.ohe_single_batch_size = ohe_single_batch_size

        # class attributes that are not parameters
        # this attribute will instantiate either
        # NA class or the GE class
        self.model = None

        # check that regression_type is valid
        check(self.regression_type in {'NA', 'GE'},
              'regression_type = %s; must be "NA", or  "GE"' %
              self.model_type)

        # choose model based on regression_type
        if regression_type == 'GE':
            self.model = GlobalEpistasisModel(self.X,
                                              self.y,
                                              self.model_type,
                                              self.test_size,
                                              self.alphabet_dict,
                                              self.ohe_single_batch_size)

            self.define_model = self.model.define_model(monotonic=self.monotonic,
                                                        custom_architecture=self.custom_architecture)

            self.model.compile_model(lr=self.learning_rate)

        elif regression_type == 'NA':
            self.model = NoiseAgnosticModel(self.X,
                                            self.y,
                                            self.model_type,
                                            self.test_size,
                                            self.alphabet_dict,
                                            self.custom_architecture,
                                            self.ohe_single_batch_size)

            self.define_model = self.model.define_model(monotonic=self.monotonic,
                                                        custom_architecture=self.custom_architecture)

            self.model.compile_model(lr=self.learning_rate)

    @handle_errors
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

        history = self.model.fit(epochs=epochs,
                                 verbose=verbose,
                                 validation_split=validation_split,
                                 use_early_stopping=use_early_stopping,
                                 early_stopping_patience=early_stopping_patience)
        return history

    @handle_errors
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
            data range which will be input to the GE nonlinearity.  If
            this is none than range will be determined from min and max
            of the latent trait.

        gauge_fix: (bool)
            if true parameters used to compute latent trait will be
            gauge fixed

        returns
        -------
        ge_nonlinearfucntion: (array-like function)
            the nonlinear GE function.

        """

        check(self.regression_type == 'GE', 'regression type must be "GE" for this function ')

        if input_range is None:
            ge_nonlinearfunction, input_range, latent_trait = self.model.ge_nonlinearity(sequences,
                                                                                         input_range=None)
            # if input range not provided, return input range
            # so ge nonliearity can be plotted against it.
            return ge_nonlinearfunction, input_range, latent_trait
        else:
            ge_nonlinearfunction = self.model.ge_nonlinearity(sequences,
                                                              input_range=input_range)
            return ge_nonlinearfunction



    @handle_errors
    def na_noisemodel(self,
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
            data range which will be input to the NA noise model.  If
            this is none than range will be determined from min and max
            of the latent trait.

        gauge_fix: (bool)
            if true parameters used to compute latent trait will be
            gauge fixed

        returns
        -------
        nanoisemodel: (array-like function)
            the noise model inferred from NA regression.

        """

        check(self.regression_type == 'NA', 'regression type must be "GE" for this function ')

        if input_range is None:
            nanoisemodel, input_range, latent_trait = self.model.noise_model(sequences,
                                                                             input_range=None)
            # if input range not provided, return input range
            # so noise model can be plotted against it.
            return nanoisemodel, input_range, latent_trait
        else:
            nanoisemodel = self.model.noise_model(sequences,
                                                  input_range=input_range)
            return nanoisemodel

    @handle_errors
    def nn_model(self):

        """
        method that returns the tf neural network
        from weights can be accessed.
        """

        return self.model.return_model()
