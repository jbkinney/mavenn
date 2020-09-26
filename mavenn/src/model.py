# Standard imports
import numpy as np
import pandas as pd
import re
import pdb
import pickle
import time
import numbers
from collections.abc import Iterable

# Scipy imports
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import lsmr

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model as kerasFunctionalModel
from tensorflow.keras.layers import Dense, Activation, Input, Lambda, Concatenate
from tensorflow.keras.constraints import non_neg as nonneg
from tensorflow.keras.callbacks import EarlyStopping

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.UI import GlobalEpistasisModel, MeasurementProcessAgnosticModel
from mavenn.src.utils import vec_data_to_mat_data
from mavenn.src.likelihood_layers import *
from mavenn.src.utils import GaussianNoiseModel, CauchyNoiseModel, SkewedTNoiseModel
from mavenn.src.entropy import mi_continuous, mi_mixed
from mavenn.src.reshape import _shape_for_output, _get_shape_and_return_1d_array, _broadcast_arrays
from mavenn.src.misc_jbk import x_to_features, x_to_stats
from mavenn.src.validate import validate_seqs, validate_1d_array, validate_alphabet
#from mavenn.src.utils import get_gpmap_params_in_cannonical_gauge

@handle_errors
class Model:

    """
    Mavenn's model class that lets the user choose either
    global epistasis regression or noise agnostic regression

    If regerssion_type == 'MPA', than ge_* parameters are not used.


    attributes
    ----------

    regression_type: (str)
        variable that choose type of regression, valid options
        include 'GE', 'MPA'

    L: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_nonlinearity_hidden_nodes:
        Number of hidden nodes (i.e. sigmoidal contributions) to use in the
        definition of the GE nonlinearity.

    ge_noise_model_type: (str)
        Specifies the type of noise model the user wants to infer.
        The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model parameters
        dependent on y_hat, and thus render the noise model heteroskedastic. Set
        to zero for a homoskedastic noise model. (Only used for GE regression).

    na_hidden_nodes:
        Number of hidden nodes (i.e. sigmoidal contributions) to use in the
        definition of the MPA measurement process.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters $\theta$.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters $\eta$.

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    Y: (int)
        Integer specifying the number of bins.
        Only used for MPA regression; set to None otherwise.

    """

    def __init__(self,
                 regression_type,
                 L,
                 alphabet,
                 gpmap_type='additive',
                 ge_nonlinearity_monotonic=True,
                 ge_nonlinearity_hidden_nodes=50,
                 ge_noise_model_type='Gaussian',
                 ge_heteroskedasticity_order=0,
                 na_hidden_nodes=50,
                 theta_regularization=0.1,
                 eta_regularization=0.1,
                 ohe_batch_size=50000,
                 Y=None):

        # Get dictionary of args passed to constructor
        # This is needed for saving models.
        self.arg_dict = locals()
        self.arg_dict.pop('self')

        # Set regression_type
        check(regression_type in {'MPA', 'GE'},
              f'regression_type = {regression_type};'
              f'must be "MPA", or "GE"')
        self.regression_type = regression_type

        # Set sequence length
        check(L > 0,
              f'len(x[0])={L}; must be > 0')
        self.L = L

        # Validate and set alphabet
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)

        # Set other parameters
        self.gpmap_type = gpmap_type
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes
        self.ge_noise_model_type = ge_noise_model_type
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.na_hidden_nodes = na_hidden_nodes
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.Y = Y

        # represents GE or MPA model object, depending which is chosen.
        # attribute value is set below
        self.model = None

        # choose model based on regression_type
        if regression_type == 'GE':

            self.model = GlobalEpistasisModel(
                            sequence_length=self.L,
                            gpmap_type=self.gpmap_type,
                            ge_nonlinearity_monotonic=
                                self.ge_nonlinearity_monotonic,
                            alphabet=self.alphabet,
                            ohe_batch_size=self.ohe_batch_size,
                            ge_heteroskedasticity_order=
                                self.ge_heteroskedasticity_order,
                            theta_regularization=self.theta_regularization,
                            eta_regularization=self.eta_regularization)

            self.define_model = self.model.define_model(
                                    ge_noise_model_type=
                                        self.ge_noise_model_type,
                                    ge_nonlinearity_hidden_nodes=
                                        self.ge_nonlinearity_hidden_nodes)

        elif regression_type == 'MPA':

            self.model = MeasurementProcessAgnosticModel(
                            sequence_length=self.L,
                            number_of_bins=self.Y,
                            alphabet=self.alphabet,
                            gpmap_type=self.gpmap_type,
                            theta_regularization=self.theta_regularization,
                            ohe_batch_size=self.ohe_batch_size)
            self.model.theta_init = None

            self.define_model = self.model.define_model(
                                    na_hidden_nodes=
                                    self.na_hidden_nodes)

    @handle_errors
    def set_data(self,
                 x,
                 y,
                 ct_n=None,
                 shuffle=True,
                 verbose=True):

        """
        Method that feeds data into the mavenn model.

        parameters
        ----------

        x: (np.ndarray)
            DNA, RNA, or protein sequences to be regressed over.

        y: (np.ndarray)
            y represents counts in bins (for MPA regression), or
            continuous measurement values (for GE regression) corresponding
            to the sequences x.

        ct_n: (np.ndarray)
            For MPA regression only. List N counts, one for each (sequence,bin)
            pair. If None, a value of 1 will be assumed for all observations.

        shuffle: (bool)
            Whether to shuffle the observations, e.g., to ensure similar
            composition of training and validation sets.

        verbose: (bool)
            Whether to provide printed feedback.

        returns
        -------
        None.
        """

        # Start timer
        set_data_start = time.time()

        # Validate x and set x
        x = validate_1d_array(x)
        x = validate_seqs(x, alphabet=self.alphabet)
        check(len(x) > 0, f'len(x)=={len(x)}; must be > 0')

        # Validate y
        y = validate_1d_array(y)

        # check that lengths are the same
        check(len(x) == len(y),
              'length of inputs (x, y) must be equal')

        # Pivot data if doing MPA regression
        if self.regression_type == 'MPA':
            self.y, self.x = vec_data_to_mat_data(x_n=x,
                                                  y_n=y,
                                                  ct_n=ct_n)
        else:
            self.x = x
            self.y = y

        # Make real sure xs are strings
        self.x = validate_seqs(self.x, alphabet=self.alphabet)

        # Set N
        self.N = len(self.x)
        if verbose:
            print(f'N = {self.N:,} observations set as training data.')

        # Shuffle data if requested
        check(isinstance(shuffle, bool),
              f"type(shuffle)={type(shuffle)}; must be bool.")
        if shuffle:
            ix = np.arange(self.N).astype(int)
            np.random.shuffle(ix)
            self.x = self.x[ix]
            if self.regression_type == 'GE':
                self.y = self.y[ix]
            else:
                self.y = self.y[ix, :]
            if verbose:
                print('Data shuffled.')

        # Check that none of the y-rows sum to zero
        # Throw an error if there are.
        if self.regression_type == 'MPA':
            num_zero_ct_rows = sum(self.y.sum(axis=1) == 0)
            check(num_zero_ct_rows == 0,
                  f'Found {num_zero_ct_rows} sequences that have no counts.'
                  f'There cannot be any such sequences.')

        # Normalize self.y -> self.y_norm
        self.y_stats = {}
        if self.regression_type == 'GE':
            y_unique = np.unique(self.y)
            check(len(y_unique),
                  f'Only {len(y_unique)} unique y-values provided;'
                  f'At least 2 are requied')
            self.y_stats['y_mean'] = self.y.mean()
            self.y_stats['y_std'] = self.y.std()

        elif self.regression_type == 'MPA':
            self.y_stats['y_mean'] = 0
            self.y_stats['y_std'] = 1

        else:
            assert False, "This shouldn't happen"

        # Set normalized y and relevant parameters
        self.y_norm = (self.y - self.y_stats['y_mean'])/self.y_stats['y_std']

        # Reshape self.y_norm to facilitate input creation
        if self.regression_type == 'GE':
            self.y_norm = np.array(self.y_norm).reshape(-1, 1)

        elif self.regression_type == 'MPA':
            self.y_norm = np.array(self.y_norm)

        # Compute the consensus sequence
        self.x_stats = x_to_stats(self.x, self.alphabet)
        self.x_ohe = self.x_stats.pop('x_ohe')
        self.x_consensus = self.x_stats['consensus_seq']
        if verbose:
            print(f'Time to set data: {time.time() - set_data_start:.3} sec.')

    @handle_errors
    def fit(self,
            epochs=50,
            learning_rate=0.005,
            validation_split=0.2,
            verbose=True,
            early_stopping=True,
            early_stopping_patience=20,
            batch_size=50,
            linear_initialization=True,
            callbacks=[],
            optimizer='Adam',
            optimizer_kwargs={},
            fit_kwargs={}):

        """
        Infers parameters, from data, for both the G-P map and the
        measurement process.

        parameters
        ----------
        epochs: (int>0)
            Maximum number of epochs to complete during training.

        learning_rate: (float > 0)
            Learning rate that will get passed to the optimizer.

        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        verbose: (0 or 1, or boolean)
            Will show training progress if 1 or True, nothing if 0 or False.

        early_stopping: (bool)
            specifies whether to use early stopping or not

        early_stopping_patience: (int)
            If using early stopping, specifies the number of epochs to wait
            after a new optimum is identified.

        batch_size: (None, int)
            Batch size to use. If None, a full-sized batch will be used.

        linear_initialization: (bool)
            Whether to initialize model with linear regression results.

        callbacks: (list)
            List of tf.keras.callbacks.Callback instances.

        optimizer: (str)
            Optimizer to use. Valid options include: ['SGD', 'RMSprop',
            'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

        optimizer_kwargs: (dict)
            Additional keyword arguments to pass to the constructor of the
            tf.keras.optimizers.Optimizer class.

        fit_kwargs: (dict):
            Additional keyword arguments to pass to tf.keras.model.fit()

        returns
        -------
        history: (tf.keras.callbacks.History object)
            Standard TensorFlow record of the optimization session.

        """

        # Start timer
        start_time = time.time()

        # Check epochs
        check(isinstance(epochs, int),
              f'type(epochs)={type(epochs)}; must be int.')
        check(epochs > 0,
              f'epochs={epochs}; must be > 0.')

        # Check learning rate & set
        check(isinstance(learning_rate, float),
              f'type(learning_rate)={type(learning_rate)}; must be float.')
        check(learning_rate > 0,
              f'learning_rate={learning_rate}; must be > 0.')
        self.learning_rate = learning_rate

        # Check epochs
        check(isinstance(validation_split, float),
              f'type(validation_split)={type(validation_split)}; '
              f'must be float.')
        check(0 < validation_split < 1,
              f'validation_split={validation_split}; must be in (0,1).')

        # Check verbose
        check(isinstance(verbose, bool),
              f'type(verbose)={type(verbose)}; must be bool.')

        # Check early_stopping
        check(isinstance(early_stopping, bool),
              f'type(early_stopping)={type(early_stopping)}; must be bool.')

        # Check early_stopping_patience
        check(isinstance(early_stopping_patience, int),
              f'type(early_stopping_patience)={type(early_stopping_patience)};'
              f' must be int.')
        check(early_stopping_patience > 0,
              f'early_stopping_patience={early_stopping_patience};'
              f'must be > 0.')

        # Check/set batch size
        check(isinstance(batch_size, (int, None)),
              f'type(batch_size)={type(batch_size)}; must be int or None.')
        if batch_size is None:
            batch_size = len(self.x)
        else:
            check(batch_size > 0,
                  f'batch_size={batch_size}; must be > 0.')

        # Check linear_initialization
        check(isinstance(linear_initialization, bool),
              f'type(linear_initialization)={type(linear_initialization)};'
              f'must be bool.')
        self.linear_initialization = linear_initialization

        # Check callbacks
        check(isinstance(callbacks, list),
              f'type(callbacks)={type(callbacks)}; must be list.')

        # Check optimizer
        check(isinstance(optimizer, str),
              f'type(optimizer)={type(optimizer)}; must be str')

        # Make Optimizer instance with specified name and learning rate
        optimizer_kwargs['learning_rate'] = learning_rate
        optimizer = tf.keras.optimizers.get({"class_name": optimizer,
                                             "config": optimizer_kwargs})

        # Check optimizer_kwargs
        check(isinstance(optimizer_kwargs, dict),
              f'type(optimizer_kwargs)={type(optimizer_kwargs)}; must be dict.')

        # Check optimizer_kwargs
        check(isinstance(fit_kwargs, dict),
              f'type(fit_kwargs)={type(fit_kwargs)}; must be dict.')

        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), \
            f'optimizer = {repr(optimizer)}' \
            'This not happen. optimizer should be ' \
            'tf.keras.optimizers.Optimizer instance by now.'

        # Compile model
        self._compile_model(optimizer=optimizer,
                            lr=self.learning_rate,
                            optimizer_kwargs=optimizer_kwargs)

        # Set early stopping callback if requested
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           mode='auto',
                                           patience=early_stopping_patience))

        # Set parameters that affect models
        self.y_mean = self.y_stats['y_mean']
        self.y_std = self.y_stats['y_std']

        # Do linear regression if requested
        if self.linear_initialization:

            # Set y targets for linear regression
            # If GE regression, use normalized y values
            if self.regression_type == 'GE':
                y_targets = self.y_norm

            # If MPA regression, use mean bin number
            elif self.regression_type == 'MPA':
                bin_nums = np.arange(self.Y)
                y_targets = (self.y_norm
                             * bin_nums[np.newaxis, :]).sum(axis=1) / \
                            self.y_norm.sum(axis=1)

            else:
                assert False, "This should never happen."

            # Do linear regression
            t = time.time()
            x_sparse = csc_matrix(self.x_ohe)
            self.theta_lc_init = lsmr(x_sparse, y_targets, show=verbose)[0]

            linear_regression_time = time.time() - t
            if verbose:
                print(f'Linear regression time: '
                      f'{linear_regression_time:.4f} sec')

            # Set weights from linear regression result
            if self.gpmap_type == 'additive':
                self.model.x_to_phi_layer.set_params(
                    theta_0=0.,
                    theta_lc=self.theta_lc_init)
            elif self.gpmap_type in ['neighbor', 'pairwise']:
                self.model.x_to_phi_layer.set_params(
                    theta_0=0.,
                    theta_lc=self.theta_lc_init,
                    theta_lclc=np.zeros([self.L, self.C, self.L, self.C]))
            else:
                assert False, "This should not happen."

        # Concatenate seqs and ys
        train_sequences = np.hstack([self.x_ohe,
                                     self.y_norm])

        # Train neural network using TensorFlow
        history = self.model.model.fit(train_sequences,
                                       self.y_norm,
                                       validation_split=validation_split,
                                       epochs=epochs,
                                       verbose=verbose,
                                       callbacks=callbacks,
                                       batch_size=batch_size,
                                       **fit_kwargs)

        # Get function representing the raw gp_map
        self._unfixed_gpmap = K.function(
            [self.model.model.layers[1].input],
            [self.model.model.layers[2].output])

        # compute unfixed phi using the function unfixed_gpmap with
        # training sequences.
        unfixed_phi = self._unfixed_gpmap([self.x_ohe])

        # Set stats
        self.unfixed_phi_mean = np.mean(unfixed_phi)
        self.unfixed_phi_std = np.std(unfixed_phi)

        # update history attribute
        self.model.history = history

        # Compute training time
        self.training_time = time.time() - start_time
        if verbose:
            print(f'Training time: {self.training_time:.1f} seconds')

        return history

    @handle_errors
    def phi_to_yhat(self,
                    phi):

        """
        Evaluate the GE nonlinearity at specified values of phi
        (the latent phenotype).

        parameters
        ----------

        phi: (array-like)
            Latent phenotype values at which to evaluate the GE nonlinearity

        returns
        -------
        y_hat: (array-like)
            GE nonlinearity evaluated on phi values

        """

        # Shape phi for processing
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # make phi unfixed
        unfixed_phi = self.unfixed_phi_mean + self.unfixed_phi_std * phi

        # Multiply by diffeomorphic mode factors
        check(self.regression_type == 'GE',
              'regression type must be "GE" for this function')

        # Compute normalized prediciton
        yhat_norm = self.model.phi_to_yhat(unfixed_phi)

        # Restore shift and scale
        yhat = self.y_mean + self.y_std * yhat_norm

        # Shape yhat for output
        yhat = _shape_for_output(yhat, phi_shape)

        return yhat

    from mavenn.src.error_handling import handle_errors, check


    @handle_errors
    def get_theta(self,
                  gauge="uniform",
                  p_lc=None,
                  x_wt=None,
                  unobserved_value=np.nan):
        """
        Returns the parameters of the G-P map and a dictionary
        of np.ndarrays.

        parameters
        ----------

        gauge: (str)
            Must be one of the following strings:
            "none" -> No gauge fixing.
            "uniform" -> Hierarchichal gauge with uniform sequence distribution.
            "empirical" -> Hierarchichal gauge with empirical sequence
                distribution from training data.
            "consensus" -> Wild-type gauge with empirical consensus sequence
                from training data.
            "user" -> Gauge set using either p_lc or x_wt supplied by user.

        p_lc: (None, array)
            An (L,C) array listing the probability of each base
            at each position. This is used when choosing the
            hierarchichal gauge. If set, must have gauge="user".

        x_wt: (str, None)
            Wild type sequence to use for gauge fixing. If set,
            must have gauge="user".

        unobserved_value: (float, None)
            Value to use for theta parameters when no corresponding
            sequences were present in the training data. If None,
            these parameters will be left alone.

        returns
        -------

        theta: (dict)
            Model parmaeters, provided as a dict of np.arrays.
        """

        # Useful alias
        _ = np.newaxis

        # Get parameters from layer
        x_stats = self.x_stats
        L = x_stats['L']
        C = x_stats['C']
        alphabet = x_stats['alphabet']
        theta_dict = self.model.x_to_phi_layer.get_params()

        # Check gauge
        choices = ("none", "uniform", "empirical", "consensus", "user")
        check(gauge in choices,
              f"Invalid choice for gauge={repr(gauge)}; "
              f"must be one of {choices}")

        # Check that p_lc is valid
        if p_lc is not None:
            check(isinstance(p_lc, np.ndarray),
                  f'type(p_lc)={type(p_lc)}; must be str.')
            check(p_lc.shape == (L, C),
                  f'p_lc.shape={p_lc.shape}; must be (L,C)={(L,C)}.')
            check(np.all(p_lc >= 0) & np.all(p_lc <= 1),
                  f'Not all p_lc values are within [0,1].')
            p_lc = p_lc / p_lc.sum(axis=1)[:, _]

        # Check that x_wt is valid
        if x_wt is not None:
            check(isinstance(x_wt, str),
                  f'type(x_wt)={type(x_wt)}; must be str.')
            check(len(x_wt) == L,
                  f'len(x_wt)={len(x_wt)}; must match L={L}.')
            check(set(x_wt) <= set(alphabet),
                  f'x_wt contains characters {set(x_wt) - set(alphabet)}'
                  f'that are not in alphabet.')

        # Check unobserved_value
        check((unobserved_value is None)
              or isinstance(unobserved_value, numbers.Number),
              f"Invalid type(unobserved_value)={type(unobserved_value)}")

        # Extract parameter arrays. Get masks and replace masked values with 0
        theta_0 = theta_dict['theta_0'].squeeze().copy()
        theta_lc = theta_dict['theta_lc'].copy()
        theta_lclc = theta_dict.get('theta_lclc',
                                    np.full(shape=(L, C, L, C),
                                            fill_value=np.nan)).copy()

        # Record nan masks and then set nan values to zero.
        nan_mask_lclc = np.isnan(theta_lclc)
        theta_lclc[nan_mask_lclc] = 0

        # Create unobserved_lc
        unobserved_lc = (x_stats['probability_df'].values == 0)

        # Set p_lc
        if gauge == "none":
            pass

        elif gauge == "uniform":
            p_lc = (1 / C) * np.ones((L, C))

        elif gauge == "empirical":
            p_lc = x_stats['probability_df'].values

        elif gauge == "consensus":
            p_lc = _x_to_mat(x_stats['consensus_seq'], alphabet)

        elif gauge == "user" and x_wt is not None:
            p_lc = _x_to_mat(x_wt, alphabet)

        elif gauge == "user" and p_lc is not None:
            pass

        else:
            assert False, 'This should not happen'

        # Fix gauge if requested
        if gauge != "none":

            # Fix 0th order parameter
            fixed_theta_0 = theta_0 \
                + np.sum(p_lc * theta_lc) \
                + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :])

            # Fix 1st order parameters
            fixed_theta_lc = theta_lc \
                - np.sum(theta_lc * p_lc, axis=1)[:, _] \
                + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                         axis=(2, 3)) \
                - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                         axis=(1, 2, 3))[:, _]

            # Fix 2nd order parameters
            fixed_theta_lclc = theta_lclc \
                - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                         axis=1)[:, _, :, :] \
                - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                         axis=3)[:, :, :, _] \
                + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                         axis=(1, 3))[:, _, :, _]

        # Otherwise, just copy over parameters
        else:
            fixed_theta_0 = theta_0
            fixed_theta_lc = theta_lc
            fixed_theta_lclc = theta_lclc

        # Set unobserved values if requested
        if unobserved_value is not None:
            # Set unobserved additive parameters
            fixed_theta_lc[unobserved_lc] = unobserved_value

            # Set unobserved pairwise parameters
            ix = unobserved_lc[:, :, _, _] | unobserved_lc[_, _, :, :]
            fixed_theta_lclc[ix] = unobserved_value

        # Set masked values back to nan
        fixed_theta_lclc[nan_mask_lclc] = np.nan

        # Create dataframe for logomaker
        logomaker_df = pd.DataFrame(index=range(L),
                                    columns=alphabet,
                                    data=fixed_theta_lc)

        # Set and return output
        theta_dict = {
            'L': L,
            'C': C,
            'alphabet': alphabet,
            'theta_0': fixed_theta_0,
            'theta_lc': fixed_theta_lc,
            'theta_lclc': fixed_theta_lclc,
            'logomaker_df': logomaker_df
        }

        # pdb.set_trace
        return theta_dict

    @handle_errors
    def get_nn(self):
        """
        Returns the tf neural network used to represent the inferred model.
        """

        return self.model.model

    @handle_errors
    def _compile_model(self,
                       optimizer,
                       lr,
                       optimizer_kwargs={},
                       compile_kwargs={}):
        """
        This method will compile the model created in the constructor.

        parameters
        ----------

        optimizer: (tf.keras.optimizers.Optimizer)
            Which optimizer to use

        lr: (float)
            Learning rate of the optimizer.

        returns
        -------
        None

        """

        # Check optimizer
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), \
            f'type(optimizer)={type(optimizer)}; must be on of ' \
            f'tf.keras.optimizers.Optimizer)'

        # Note: this loss just returns the computed
        # Likelihood in the custom likelihood layer
        def likelihood_loss(y_true, y_pred):
            if self.regression_type == 'GE':
                return K.sum(y_pred)
            elif self.regression_type == 'MPA':
                return y_pred

        self.model.model.compile(loss=likelihood_loss,
                                 optimizer=optimizer,
                                 **compile_kwargs)

    @handle_errors
    def x_to_phi(self, x):
        """

        Evaluates the latent phenotype phi on input sequences.

        parameters
        ----------
        x: (array-like of str)
            Sequence inputs representing DNA, RNA, or protein (whichever
            type of sequence the model was trained on). Input can must be
            an array of str, all the proper length.

        returns
        -------
        phi: (array-like of float)
            Array of latent phenotype values.

        """

        # Shape x for processing
        x, x_shape = _get_shape_and_return_1d_array(x)

        # Check seqs
        x = validate_seqs(x, alphabet=self.alphabet)
        check(len(x[0]) == self.L,
              f'len(x[0])={len(x[0])}; should be L={self.L}')

        # Encode sequences as features
        stats = x_to_stats(x=x, alphabet=self.alphabet)
        x_ohe = stats.pop('x_ohe')

        # Keras function that computes phi from x
        gpmap_function = K.function([self.model.model.layers[1].input],
                                    [self.model.model.layers[2].output])

        # Compute latent phenotype values
        # Note that these are NOT diffeomorphic-mode fixed
        unfixed_phi = gpmap_function([x_ohe])

        # Fix diffeomorphic models
        phi = (unfixed_phi - self.unfixed_phi_mean) / self.unfixed_phi_std

        # Shape phi for output
        phi = _shape_for_output(phi, x_shape)

        # Return latent phenotype values
        return phi


    @handle_errors
    def x_to_yhat(self,
                  x):
        """
        Make predictions for arbitrary input sequences. Note that this returns
        the output of the measurement process, not the latent phenotype.

        parameters
        ----------
        x: (array-like)
            Sequence data on which to make predictions.

        returns
        -------
        predictions: (array-like)
            An array of predictions for GE regression.
        """

        # Shape x for processing
        x, x_shape = _get_shape_and_return_1d_array(x)

        check(self.regression_type == 'GE',
              'Regression type must be GE for this function.')

        yhat = self.phi_to_yhat(self.x_to_phi(x))

        # Shape yhat for output
        yhat = _shape_for_output(yhat, x_shape)

        return yhat

    def I_predictive(self,
                     x,
                     y,
                     knn=5,
                     uncertainty=True,
                     num_subsamples=25,
                     use_LNC=False,
                     alpha_LNC=.5,
                     verbose=False):
        """
        Estimate the predictive information I[y;phi] on supplied data.

        parameters
        ----------

        x: (array-like of strings)
            Array of sequences for which to comptue phi values.

        y: (array-like of floats)
            Array of measurements y to use when computing I[y;phi].
            If measurements are continuous, y must be the same shape as
            x. If measurements are discrete, y can have two formats.
            If y_format="list", y should be a list of discrete values,
            one for each x. If y_format="matrix", y should be a
            MxY matrix, where M=len(x) and Y is the number of possible
            values for Y.

        knn: (int>0)
            Number of nearest neighbors to use in the KSG estimator.

        uncertainty: (bool)
            Whether to estimate the uncertainty of the MI estimate.
            Substantially increases runtime if True.

        num_subsamples: (int > 0)
            Number of subsamples to use if estimating uncertainty.

        use_LNC: (bool)
            Whether to compute the Local Nonuniform Correction
            (LNC) using the method of Gao et al., 2015.
            Substantially increases runtime if True. Only used for
            continuous y values.

        alpha_LNC: (float in (0,1))
            Value of alpha to use when computing LNC.
            See Gao et al., 2015 for details. Only used for
            continuous y values.

        verbose: (bool)
            Whether to print results and execution time.

        returns
        -------

        (I, dI): (float, float)
            I = Mutual information estimate in bits.
            dI = Uncertainty estimate in bits. Zero if uncertainty=False is set.
            Not returned if uncertainty=False is set.
        """

        if self.regression_type=='GE':
            return mi_continuous(self.x_to_phi(x),
                                 y,
                                 knn=knn,
                                 uncertainty=uncertainty,
                                 use_LNC=use_LNC,
                                 alpha_LNC=alpha_LNC,
                                 verbose=verbose)

        elif self.regression_type=='MPA':
            phi = self.x_to_phi(x)
            return mi_mixed(phi,
                            y,
                            knn=knn,
                            uncertainty=uncertainty,
                            num_subsamples=num_subsamples,
                            verbose=verbose)

    def yhat_to_yq(self,
                   yhat,
                   q=[0.16, 0.84]):
        """
        Returns quantile values of p(y|yhat) given yhat and the quantiles q.
        Reserved only for GE models

        parameters
        ----------

        yhat: (array of floats)
            Values from which p(y|yhat) is computed.

        q: (array of floats in [0,1])
            Quantile specifications

        returns
        -------

        yq: (array of floats)
            Array of quantile values.
        """

        # Shape yhat for processing
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)
        yhat_norm = (yhat - self.y_mean)/self.y_std

        # Shape x for processing
        q, q_shape = _get_shape_and_return_1d_array(q)

        # Make sure this is the right type of model
        check(self.regression_type=='GE',
              'regression type must be GE for this methdd')

        # Get GE noise model based on the users input.
        yq_norm = globals()[self.ge_noise_model_type + 'NoiseModel']\
            (self, yhat_norm, q=q).user_quantile_values

        # This seems to be needed
        yq_norm = np.array(yq_norm).T

        # Restore scale and shift
        yq = self.y_mean + self.y_std * yq_norm

        # Shape yqs for output
        yq_shape = yhat_shape + q_shape
        yq = _shape_for_output(yq, yq_shape)

        return yq

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Computes the p(y|phi) for both GE and MPA regression.

        y: (number or array-like of numbers)
            Measurement values. Note that these are cast as integers for
            MPA regression.

        phi: (float or array-like of floats)
            Latent phenotype values.

        paired: (bool)
            Whether y,phi values should be treated as pairs.
            If so, y and phi must have the same number of elements.
            The shape of y will be used as output.

        returns
        -------
            p: (float or array-like of floats)
                Probability of y given phi.
        """

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # Normalize y values
        y_norm = (y - self.y_mean)/self.y_std

        # Unfix phi value
        unfixed_phi = self.unfixed_phi_mean + self.unfixed_phi_std * phi

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Do computation
            p_norm = self._p_of_y_given_phi(y_norm, unfixed_phi)

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y_norm, unfixed_phi = _broadcast_arrays(y_norm, unfixed_phi)

            # Do computation
            p_norm = self._p_of_y_given_phi(y_norm, unfixed_phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # De-normalize probability
        p = p_norm / self.y_std

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    # TODO: Stated behavior won't work for MPA regression, only GE
    def _p_of_y_given_phi(self,
                         y,
                         phi):

        """
        Method that computes the p(y|phi) for both GE and MPA regression.

        Note that if y is and np.ndarray with shape=(n1,n2,...,nK) and
        phi is an np.ndarray with shape=(n1,n2,...,nK), the returned value
        p_of_y_given_phi will also have shape=(n1,n2,...,nK). In other
        cases, the appropriate broadcasting will occur.

        y: (float (GE) or int (MPA))
            Specifies continuous target value for GE regression or an integer
            specifying bin number for MPA regression.

        phi: (float)
            Latent phenotype on which probability is conditioned.

        returns
        -------
        p_of_y_given_phi: (float)
            Probaility of y given phi.

        """

        if self.regression_type == 'MPA':

            in_shape = y.shape

            # Cast y as integers
            y = y.astype(int)

            # Make sure all y values are valid
            check(np.all(y >= 0),
                  f"Negative values for y are invalid for MAP regression")

            check(np.all(y < self.Y),
                  f"Some y values exceed the number of bins {self.Y}")

            # Have to ravel
            y = y.ravel()
            phi = phi.ravel()

            # Get values for all bins
            p_of_all_y_given_phi = self.na_p_of_all_y_given_phi(phi)

            # There has to be a better way to do this
            p_of_y_given_phi = np.zeros(len(y))
            for i, _y in enumerate(y):
                p_of_y_given_phi[i] = p_of_all_y_given_phi[i, _y]

            # Reshape
            p_of_y_given_phi = np.reshape(p_of_y_given_phi, in_shape)

        else:
            check(self.regression_type=='GE',
                  f'Invalid regression type {self.regression_type}.')

            # variable to store the shape of the returned object
            yhat = self.phi_to_yhat(np.array(phi).ravel())
            if np.array(y).shape==np.array(phi).shape \
                    and (len(np.array(y).shape)>0 \
                         and len(np.array(phi).shape)>0):
                shape = np.array(y).shape
                p_of_y_given_phi = self._p_of_y_given_y_hat(y.ravel(),
                                       yhat.ravel()).reshape(shape)

            else:
                p_of_y_given_phi = self._p_of_y_given_y_hat(y, yhat)

        return p_of_y_given_phi

    def p_of_y_given_yhat(self, y, yhat, paired=False):
        """
        Computes the p(y|yhat) for GE only.

        y: (float or array-like of floats)
            Measurement values.

        yhat: (float or array-like of floats)
            Latent phenotype values.

        paired: (bool)
            Whether y,yhat values should be treated as pairs.
            If so, y and yhat must have the same number of elements.
            The shape of y will be used as output.

        returns
        -------
            p: (float or array-like of floats)
                Probability of y given yhat.
        """

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)

        # Normalize y values
        y_norm = (y - self.y_mean)/self.y_std
        yhat_norm = (yhat - self.y_mean)/self.y_std

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == yhat_shape,
                  f"y shape={y_shape} does not match yhat shape={yhat_shape}")

            # Do computation
            p_norm = self._p_of_y_given_y_hat(y_norm, yhat_norm)

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and yhat
            y_norm, yhat_norm = _broadcast_arrays(y_norm, yhat_norm)

            # Do computation
            p_norm = self._p_of_y_given_y_hat(y_norm, yhat_norm)

            # Set output shape
            p_shape = y_shape + yhat_shape

        # De-normalize probability
        p = p_norm / self.y_std

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    def _p_of_y_given_y_hat(self,
                            y,
                            yhat):

        """
        Method that returns computes.

        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.

        yhat: (float)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p_of_y_given_yhat: (array-like of floats)
            Probability of y given sequence yhat. Shape of returned value will
            match shape of y_test, for a single yhat. For each value of yhat_i,
            the distribution p(y|yhat_i), where i traverses the elements of
            yhat.

        """

        check(self.regression_type=='GE',
              "This method works on with GE regression.")

        # Get GE noise model based on the users input.
        ge_noise_model = globals()[self.ge_noise_model_type +
                                   'NoiseModel'](self, yhat, None)

        return ge_noise_model.p_of_y_given_yhat(y, yhat)

    def __p_of_y_given_y_hat(self,
                             y,
                             yhat):
        """
        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.
            y: np.ndarray, shape=(n1,n2,...,nK

        yhat: (array-like of floats)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p: np.ndarray, shape=(n1,n2,...,nK)

        """

        check(self.regression_type=='GE',
              "This method works on with GE regression.")

        # Get GE noise model based on the users input.
        ge_noise_model = globals()[self.ge_noise_model_type +
                                   'NoiseModel'](self, yhat, None)
        vec_p_of_y_given_yhat = np.vectorize(ge_noise_model.p_of_y_given_yhat)

        return vec_p_of_y_given_yhat(y, yhat)

    def na_p_of_all_y_given_phi(self,
                                phi):

        """
        Evaluate the MPA measurement process at specified values of
        phi (the latent phenotype).

        parameters
        ----------
        phi: (array-like)
            Latent phenotype values at which to evaluate the measurement
            process.

        returns
        -------
        p_of_dot_given_phi: (array-like)
            Measurement process p(y|phi) for all possible values of y. Is of
            size MxY where M=len(phi) and Y is the number of possible y values.

        """

        check(self.regression_type == 'MPA',
              'regression type must be "MPA" for this function ')

        p_of_dot_given_phi = self.model.p_of_all_y_given_phi(phi)

        return p_of_dot_given_phi

    def p_of_y_given_x(self, y, x):

        """
        Method that computes p_of_y_given_x.

        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.

        x: (array-like of strings)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p_of_y_given_x: (array-like of floats)
            Probability of y given sequence x. Shape of returned value will
            match shape of y_test.

        """


        if self.regression_type=='GE':
            yhat = self.x_to_yhat(x)
            # Get GE noise model based on the users input.
            ge_noise_model = globals()[self.ge_noise_model_type
                                       + 'NoiseModel'](self,yhat)

            p_of_y_given_x = ge_noise_model.p_of_y_given_yhat(y, yhat)
            return p_of_y_given_x

        elif self.regression_type=='MPA':

            # check that entered y (specifying bin number) is an integer
            check(isinstance(y, int),
                  'type(y), specifying bin number, must be of type int')

            # check that entered bin nnumber doesn't exceed max bins
            check(y < self.y_norm[0].shape[0],
                  "bin number cannot be larger than max bins = %d" %
                  self.y_norm[0].shape[0])

            phi = self.x_to_phi(x)
            p_of_y_given_x = self.p_of_y_given_phi(y, phi)
            return p_of_y_given_x

    def save(self,
             filename,
             verbose=True):

        """
        Method that will save the MAVE-NN model. Note: this does NOT
        save training data

        parameters
        ----------
        filename: (str)
            filename of the saved model.

        verbose: (bool)
            Whether to provide user feedback.

        returns
        -------
        None

        """

        # Create config_dict
        config_dict = {
            'model_kwargs': self.arg_dict,
            'unfixed_phi_mean': self.unfixed_phi_mean,
            'unfixed_phi_std': self.unfixed_phi_std,
            'y_std': self.y_std,
            'y_mean': self.y_mean,
            'x_stats': self.x_stats,
            'y_stats': self.y_stats
        }

        # Save config_dict as pickle file
        filename_pickle = filename + '.pickle'
        with open(filename_pickle, 'wb') as f:
            pickle.dump(config_dict, f)

        # save weights
        filename_h5 = filename + '.h5'
        self.get_nn().save_weights(filename_h5)

        if verbose:
            print(f'Model saved to these files:\n'
                  f'\t{filename_pickle}\n'
                  f'\t{filename_h5}')

# Converts sequences to matrices
def _x_to_mat(x, alphabet):
    return (np.array(list(x))[:, np.newaxis] ==
            alphabet[np.newaxis, :]).astype(float)