"""model.py: Define the Model() class, which represents all MAVE-NN models."""
# Standard imports
import numpy as np
import pandas as pd
import pdb
import pickle
import time
import numbers

# Scipy imports
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr
from scipy.stats import spearmanr

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

# sklearn import
import sklearn.preprocessing

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.regression_types import GlobalEpistasisModel, \
                                        MeasurementProcessAgnosticModel
from mavenn.src.entropy import mi_continuous, mi_mixed, entropy_continuous
from mavenn.src.reshape import _shape_for_output, \
                               _get_shape_and_return_1d_array, \
                               _broadcast_arrays
from mavenn.src.validate import validate_seqs, \
                                validate_1d_array, \
                                validate_alphabet
from mavenn.src.utils import mat_data_to_vec_data, \
                             vec_data_to_mat_data, \
                             x_to_stats, \
                             p_lc_to_x, _x_to_mat


class Model:
    """
    Represents a MAVE-NN model.

    Parameters
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
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    na_hidden_nodes:
        Number of hidden nodes (i.e. sigmoidal contributions) to use in the
        definition of the MPA measurement process.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    Y: (int)
        Integer specifying the number of bins.
        Only used for MPA regression; set to None otherwise.
    """

    @handle_errors
    def __init__(self,
                 regression_type,
                 L,
                 alphabet,
                 gpmap_type='additive',
                 ge_nonlinearity_type='nonlinear',
                 ge_nonlinearity_monotonic=True,
                 ge_nonlinearity_hidden_nodes=50,
                 ge_noise_model_type='Gaussian',
                 ge_heteroskedasticity_order=0,
                 na_hidden_nodes=50,
                 theta_regularization=0.1,
                 eta_regularization=0.1,
                 ohe_batch_size=50000,
                 Y=None):
        """Model() class constructor."""
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
        self.ge_nonlinearity_type = ge_nonlinearity_type
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes
        self.ge_noise_model_type = ge_noise_model_type
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.na_hidden_nodes = na_hidden_nodes
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.Y = Y

        # Variables needed for saving
        self.unfixed_phi_mean = np.nan
        self.unfixed_phi_std = np.nan
        self.y_std = np.nan
        self.y_mean = np.nan
        self.x_stats = {}
        self.y_stats = {}
        self.history = {}

        # Dictionary to pass information to layers
        self.info_for_layers_dict = {'H_y': np.nan,
                                     'H_y_norm': np.nan,
                                     'dH_y': np.nan}

        # represents GE or MPA model object, depending which is chosen.
        # attribute value is set below
        self.model = None

        # choose model based on regression_type
        if regression_type == 'GE':

            self.model = GlobalEpistasisModel(
                            info_for_layers_dict=self.info_for_layers_dict,
                            sequence_length=self.L,
                            gpmap_type=self.gpmap_type,
                            ge_nonlinearity_type=self.ge_nonlinearity_type,
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

            # Set layers
            self.layer_gpmap = self.model.x_to_phi_layer
            self.layer_nonlinearity = self.model.phi_to_yhat_layer
            self.layer_noise_model = self.model.noise_model_layer

        elif regression_type == 'MPA':

            self.model = MeasurementProcessAgnosticModel(
                            info_for_layers_dict=self.info_for_layers_dict,
                            sequence_length=self.L,
                            number_of_bins=self.Y,
                            alphabet=self.alphabet,
                            gpmap_type=self.gpmap_type,
                            theta_regularization=self.theta_regularization,
                            eta_regularization=self.eta_regularization,
                            ohe_batch_size=self.ohe_batch_size)
            self.model.theta_init = None

            self.define_model = self.model.define_model(
                                    na_hidden_nodes=
                                    self.na_hidden_nodes)

            # Set layers
            self.layer_gpmap = self.model.x_to_phi_layer
            self.layer_measurement_process = \
                self.model.layer_measurement_process


    @handle_errors
    def set_data(self,
                 x,
                 y,
                 ct=None,
                 shuffle=True,
                 verbose=True):
        """
        Set training data.

        Parameters
        ----------
        x: (np.ndarray)
            1D array of N sequences, each of the same length.

        y: (np.ndarray)
            Array of measurements.
            For GE regression, y must be a 1D array of floats, length N.
            For MPA regression, y must be a 1D or 2D array of nonnegative ints.
                - If 1D, will be interpretd as listing bin numbers, and
                    must be of length N.
                - If 2D, will be interpreted as listing counts across all bins,
                    and must be of shape (N,Y) where Y is the number of bins

        ct: (np.ndarray or None)
            Only used for MPA regression when y is 1D. In this case, represents
            the number of observations of each sequence in each bin. Must
            then be 1D array, length N, of nonnegative ints.

        shuffle: (bool)
            Whether to shuffle the observations, e.g., to ensure similar
            composition of training and validation sets.

        verbose: (bool)
            Whether to provide printed feedback.

        Returns
        -------
        None
        """
        # Start timer
        set_data_start = time.time()

        # Validate x and set x
        x = validate_1d_array(x)
        x = validate_seqs(x, alphabet=self.alphabet)
        check(len(x) > 0, f'len(x)=={len(x)}; must be > 0')

        # Validate y, note that this doesn't
        # apply for MPA regression since y
        # is not a 1-d array in MPAR.
        if self.regression_type == 'GE':
            y = validate_1d_array(y)
            check(len(x) == len(y), 'length of inputs (x, y) must be equal')

        elif self.regression_type == 'MPA':
            if y.ndim == 1:
                y, x = vec_data_to_mat_data(y_n=y, ct_n=ct, x_n=x)
            else:
                check(y.ndim == 2,
                      f'y.ndim={y.ndim}; must be 1 or 2.')

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
            self.y_std = self.y.std()
            self.y_mean = self.y.mean()
            self.y_stats['y_mean'] = self.y_mean
            self.y_stats['y_std'] = self.y_std

        elif self.regression_type == 'MPA':
            self.y_std = 1
            self.y_mean = 0
            self.y_stats['y_mean'] = self.y_mean
            self.y_stats['y_std'] = self.y_std

        else:
            assert False, "This shouldn't happen"

        # Set normalized y and relevant parameters
        self.y_norm = (self.y - self.y_stats['y_mean'])/self.y_stats['y_std']

        # Reshape self.y_norm to facilitate input creation
        if self.regression_type == 'GE':
            self.y_norm = np.array(self.y_norm).reshape(-1, 1)

            # Subsample y_norm for entropy estimation if necessary
            N_max = int(1E4)
            if self.N > N_max:
                z = np.random.choice(a=self.y_norm.squeeze(),
                                     size=N_max,
                                     replace=False)
            else:
                z = self.y_norm.squeeze()

            # Add some noise to aid in entropy estimation
            z += 1E-3 * np.random.randn(z.size)

            # Compute entropy
            H_y_norm, dH_y = entropy_continuous(z, knn=7)
            H_y = H_y_norm + np.log2(self.y_std)

            self.info_for_layers_dict['H_y'] = H_y
            self.info_for_layers_dict['H_y_norm'] = H_y_norm
            self.info_for_layers_dict['dH_y'] = dH_y

        elif self.regression_type == 'MPA':
            self.y_norm = np.array(self.y_norm)

            # Compute naive entropy estimate
            # Should probably be OK in most cases
            # Ideally we'd use the NSB estimator
            c_y = self.y_norm.sum(axis=0).squeeze()
            p_y = c_y / c_y.sum()
            ix = p_y > 0
            H_y_norm = -np.sum(p_y[ix] * np.log2(p_y[ix]))
            H_y = H_y_norm + np.log2(self.y_std)
            dH_y = 0 # Need NSB to estimate this well
            self.info_for_layers_dict['H_y'] = H_y
            self.info_for_layers_dict['H_y_norm'] = H_y_norm
            self.info_for_layers_dict['dH_y'] = dH_y

        # Compute sequence statistics
        self.x_stats = x_to_stats(self.x, self.alphabet)

        # Extract one-hot encoding of sequences
        # This is what is passed to the network.
        self.x_ohe = self.x_stats.pop('x_ohe')

        # Extract consensus sequence
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
        Infer values for model parameters.

        Parameters
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

        Returns
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

        # Check optimizer_kwargs
        check(isinstance(optimizer_kwargs, dict),
              f'type(optimizer_kwargs)={type(optimizer_kwargs)}; must be dict.')

        # Make Optimizer instance with specified name and learning rate
        optimizer_kwargs['learning_rate'] = learning_rate
        optimizer = tf.keras.optimizers.get({"class_name": optimizer,
                                             "config": optimizer_kwargs})

        # Check optimizer_kwargs
        check(isinstance(fit_kwargs, dict),
              f'type(fit_kwargs)={type(fit_kwargs)}; must be dict.')


        # Returns the sum of negative log likelihood contributions
        # from each sequence, which is provided as y_pred
        def likelihood_loss(y_true, y_pred):
            return K.sum(y_pred)

        self.model.model.compile(loss=likelihood_loss,
                                 optimizer=optimizer)

        # Set early stopping callback if requested
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           mode='auto',
                                           patience=early_stopping_patience))

        # Set parameters that affect models
        self.y_mean = self.y_stats['y_mean']
        self.y_std = self.y_stats['y_std']

        # Set y targets for linear regression and sign assignment
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

        # Do linear regression if requested
        if self.linear_initialization:

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
        unfixed_phi = self._unfixed_gpmap(self.x_ohe)[0].ravel()

        # Set stats
        self.unfixed_phi_mean = np.mean(unfixed_phi)
        self.unfixed_phi_std = np.std(unfixed_phi)

        # Flip sign if correlation of phi with y_targets is negative
        r, p_val = spearmanr(unfixed_phi, y_targets)
        if r < 0:
            self.unfixed_phi_std *= -1.

        # update history attribute
        self.history = history.history

        # Compute training time
        self.training_time = time.time() - start_time
        if verbose:
            print(f'Training time: {self.training_time:.1f} seconds')

        return history

    @handle_errors
    def phi_to_yhat(self,
                    phi):
        """
        Map latent phenotype (phi) values to observable (yhat) values.

        Only used for GE regression models.

        Parameters
        ----------
        phi: (array-like)
            Latent phenotype values at which to evaluate the GE nonlinearity.

        Returns
        -------
        y_hat: (array-like)
            Observable values.
        """
        # Shape phi for processing
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # make phi unfixed
        unfixed_phi = self.unfixed_phi_mean + self.unfixed_phi_std * phi

        # Multiply by diffeomorphic mode factors
        check(self.regression_type == 'GE',
              'regression type must be "GE" for this function')

        # Compute normalized phi using nonlinearity layer
        yhat_norm = self.layer_nonlinearity.phi_to_yhat(unfixed_phi,
                                                        use_arrays=True)

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
        Get parameters (theta) of the G-P map.

        Parameters
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

        Returns
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

        return theta_dict

    @handle_errors
    def get_nn(self):
        """Return the model's TensorFlow neural network backend."""
        return self.model.model

    @handle_errors
    def x_to_phi(self, x):
        """
        Compute the latent phenotype (phi) from sequences (x).

        Parameters
        ----------
        x: (array-like of str)
            Sequence inputs representing DNA, RNA, or protein (whichever
            type of sequence the model was trained on). Input can must be
            an array of str, all the proper length.

        Returns
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
        Map sequences (x) to observables (yhat).

        Parameters
        ----------
        x: (array-like)
            Sequence data on which to make predictions.

        Returns
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

    @handle_errors
    def simulate_dataset(self,
                         N,
                         training_frac=.8):
        """
        Simulate a dataset based on the MAVE-NN model.

        Parameters
        ----------
        N: (int > 0)
            The number of observations to simulate.

        training_frac: (float in [0,1])
            The fraction of sequences to label for training.

        Returns
        -------
        data_df: (pd.DataFrame)
            Simulated dataset formatted as a dataframe. Columns include
            'training_set', 'phi', 'y', 'x'. If model is GE, an additional
            column 'yhat' is added. If model is MPA, an additional column
            'ct' is added. Note that, under MPA regression, N will be the
            sum of values in the 'ct' column, not the number of rows in the
            dataframe.
        """
        # Validate N
        check(isinstance(N, int),
              f'type(N)={type(N)}; must be int.')
        check(N > 0,
              f'N={N}; must be > 0')

        # Validate training_frac
        check(isinstance(training_frac, float),
                         f'type(training_frac)={type(training_frac)};'
                         'must be float.')
        check(0 <= training_frac <= 1,
              f'training_frac={training_frac}; must be in [0,1]')

        # Generate sequences
        x = p_lc_to_x(N=N,
                      p_lc=self.x_stats['probability_df'].values,
                      alphabet=self.x_stats['alphabet'])

        # Compute phi values
        phi = self.x_to_phi(x)

        if self.regression_type == 'MPA':

            # Compute grid of p(y|\phi) values over all y for all phi
            all_y = np.arange(self.Y).astype(int)
            p_all_y_given_phi = self.p_of_y_given_phi(all_y,
                                                      phi,
                                                      paired=False).T

            # Create function to choose y
            def choose_y(p_all_y):
                return np.random.choice(a=all_y,
                                        size=1,
                                        replace=True,
                                        p=p_all_y)

            # Choose y values
            y = np.apply_along_axis(choose_y, axis=1, arr=p_all_y_given_phi)

            # Get counts in bins. LabelBinarizer is efficient,
            # e.g. N = 10^5 takes ~ 0.01 seconds.
            label_binarizer = sklearn.preprocessing.LabelBinarizer()
            label_binarizer.fit(range(self.Y))
            ct_ = label_binarizer.transform(y)

        elif self.regression_type == 'GE':

            # Compute yhat
            yhat = self.phi_to_yhat(phi)

            # Normalize yhat
            yhat_norm = (yhat - self.y_mean)/self.y_std

            # Get layer
            layer = self.layer_noise_model

            # Sample values
            y_norm = layer.sample_y_given_yhat(yhat_norm)

            # Compute y from y_norm
            y = self.y_mean + self.y_std * y_norm

        else:
            assert False, 'This should not happen.'

        # Store results in dataframe and return
        data_df = pd.DataFrame()
        data_df['phi'] = phi
        if self.regression_type == 'GE':
            data_df['y'] = y
        data_df['x'] = x

        # If doing MPA regression, collapse by sequence and add ct col
        if self.regression_type == 'MPA':

            # merge the y counts array with the dataframe created above
            # and name bin columns with prefix 'ct_*'
            y_df = pd.DataFrame(data=ct_,
                                columns=['ct_' + str(n) for n in range(self.Y)])
            data_df = pd.concat([data_df, y_df], axis=1)

        elif self.regression_type == 'GE':
            data_df.insert(0, column='yhat', value=yhat)

        # Choose training frac
        train_ix = np.random.rand(len(data_df)) < training_frac
        data_df.insert(0, column='training_set', value=train_ix)

        return data_df

    @handle_errors
    def I_likelihood(self,
                     x,
                     y,
                     ct=None,
                     uncertainty=True):
        """
        Estimate the likelihood information (I_like) on user-provided data.

        Parameters
        ----------
        x: (np.ndarray)
            1D array of N sequences, each of the same length.

        y: (np.ndarray)
            Array of measurements.
            For GE regression, y must be a 1D array of floats, length N.
            For MPA regression, y must be a 1D or 2D array of nonnegative ints.
                - If 1D, will be interpretd as listing bin numbers, and
                    must be of length N.
                - If 2D, will be interpreted as listing counts across all bins,
                    and must be of shape (N,Y) where Y is the number of bins

        ct: (np.ndarray or None)
            Only used for MPA regression when y is 1D. In this case, represents
            the number of observations of each sequence in each bin. Must
            then be 1D array, length N, of nonnegative ints.

        uncertainty: (bool)
            Whether to estimate the uncertainty of the MI estimate.

        Returns
        -------
        (I, dI): (float, float)
            I = Mutual information estimate in bits.
            dI = Uncertainty estimate in bits. Zero if uncertainty=False is set.
        """
        if self.regression_type == 'GE':

            # Number of datapoints
            N = len(y)

            # Normalize y values
            y_norm = (y - self.y_mean) / self.y_std

            # Subsample y_norm for entropy estimation if necessary
            N_max = int(1E4)
            if N > N_max:
                z = np.random.choice(a=y_norm.squeeze(),
                                     size=N_max,
                                     replace=False)
            else:
                z = y_norm.squeeze()

            # Add some noise to aid in entropy estimation
            z += 1E-3 * np.random.randn(z.size)

            # Compute entropy
            H_y_norm, dH_y = entropy_continuous(z, knn=7)
            H_y = H_y_norm + np.log2(self.y_std)

            # Compute phi
            phi = self.x_to_phi(x)

            # Compute p_y_give_phi
            p_y_given_phi = self.p_of_y_given_phi(y,
                                                  phi,
                                                  paired=True)

            # Compute H_y_given_phi
            H_y_given_phi_n = -np.log2(p_y_given_phi)

        elif self.regression_type == 'MPA':

            # If y is 2D, convert from mat data to vec data
            if y.ndim == 2:
                y, ct, x = mat_data_to_vec_data(ct_my=y, x_m=x)

            # If ct is not set, set to ones
            if ct is None:
                ct = np.ones(y.size)

            # Expand x and y based on ct values
            y = np.concatenate(
                        [[y_n]*ct_n for y_n, ct_n in zip(y, ct)])
            x = np.concatenate(
                        [[x_n]*ct_n for x_n, ct_n in zip(x, ct)])

            # Number of datapoints
            ct_y = np.array([(y==i).sum() for i in range(self.Y)])
            p_y = ct_y / ct_y.sum()
            ix = p_y > 0
            H_y_norm = -np.sum(p_y[ix] * np.log2(p_y[ix]))
            H_y = H_y_norm + np.log2(self.y_std)
            dH_y = 0  # Need NSB to estimate this well

            # Compute phi
            phi = self.x_to_phi(x)

            p_y_given_phi = self.p_of_y_given_phi(y, phi, paired=True)
            H_y_given_phi_n = -np.log2(p_y_given_phi)

        # Get total number of independent observations
        N = len(H_y_given_phi_n)

        # Compute H_y_given_phi
        H_y_given_phi = np.mean(H_y_given_phi_n)

        # Compute uncertainty
        dH_y_given_phi = np.std(H_y_given_phi_n, ddof=1)/np.sqrt(N)

        # Compute I_like and dI_fit
        I_like = H_y - H_y_given_phi
        if uncertainty:
            dI_like = np.sqrt(dH_y**2 + dH_y_given_phi**2)
        else:
            dI_like = 0

        return I_like, dI_like

    @handle_errors
    def I_predictive(self,
                     x,
                     y,
                     ct=None,
                     knn=5,
                     uncertainty=True,
                     num_subsamples=25,
                     use_LNC=False,
                     alpha_LNC=.5,
                     verbose=False):
        """
        Estimate the predictive information I_pred on user-provided data.

        Parameters
        ----------
        x: (np.ndarray)
            1D array of N sequences, each of the same length.

        y: (np.ndarray)
            Array of measurements.
            For GE regression, y must be a 1D array of floats, length N.
            For MPA regression, y must be a 1D or 2D array of nonnegative ints.
                - If 1D, will be interpretd as listing bin numbers, and
                    must be of length N.
                - If 2D, will be interpreted as listing counts across all bins,
                    and must be of shape (N,Y) where Y is the number of bins

        ct: (np.ndarray or None)
            Only used for MPA regression when y is 1D. In this case, represents
            the number of observations of each sequence in each bin. Must
            then be 1D array, length N, of nonnegative ints.

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

        Returns
        -------
        I: (float)
            Mutual information estimate in bits.
        dI: (float)
            Uncertainty estimate in bits. Zero if uncertainty=False is set.
        """
        if self.regression_type == 'GE':

            return mi_continuous(self.x_to_phi(x),
                                 y,
                                 knn=knn,
                                 uncertainty=uncertainty,
                                 use_LNC=use_LNC,
                                 alpha_LNC=alpha_LNC,
                                 verbose=verbose)

        elif self.regression_type == 'MPA':

            # If y is 2D, convert from mat data to vec data
            if y.ndim == 2:
                y, ct, x = mat_data_to_vec_data(ct_my=y, x_m=x)

            # If ct is not set, set to ones
            if ct is None:
                ct = np.ones(y.size)

            # Expand x and y based on ct values
            y = np.concatenate(
                        [[y_n]*ct_n for y_n, ct_n in zip(y, ct)])
            x = np.concatenate(
                        [[x_n]*ct_n for x_n, ct_n in zip(x, ct)])

            # Compute phi
            phi = self.x_to_phi(x)

            # Add random component to phi to regularize information estimate
            phi += (1.0E-3) * np.random.randn(len(x))

            # Compute mi_mixed on expanded y and phi
            return mi_mixed(phi,
                            y,
                            knn=knn,
                            uncertainty=uncertainty,
                            num_subsamples=num_subsamples,
                            verbose=verbose)

    def yhat_to_yq(self,
                   yhat,
                   q=[0.16, 0.84], paired=False):
        """
        Return quantile values of p(y|yhat). Used only for GE models.

        Parameters
        ----------
        yhat: (array of floats)
            Values from which p(y|yhat) is computed.

        q: (array of floats in [0,1])
            Quantile specifications

        paired: (bool)
            Whether yhat, q values should be treated as pairs.
            If so, yhat and q must have the same number of elements.
            The shape of yhat will be used as output.

        Returns
        -------
        yq: (array of floats)
            Array of quantile values.
        """
        # Prepare inputs
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)
        q, q_shape = _get_shape_and_return_1d_array(q)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(yhat_shape == q_shape,
                  f"yhat shape={yhat_shape} does not "
                  f"match q shape={q_shape}")

            # Use y_shape as output shape
            yq_shape = yhat_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            yhat, q = _broadcast_arrays(yhat, q)

            # Set output shape
            yq_shape = yhat_shape + q_shape


        # Make sure this is the right type of model
        check(self.regression_type=='GE',
              'regression type must be GE for this methdd')

        # Normalize yhat
        yhat_norm = (yhat - self.y_mean) / self.y_std

        # Get layer
        layer = self.layer_noise_model

        # Use layer to compute normalized quantile
        yq_norm = layer.yhat_to_yq(yhat=yhat_norm, q=q, use_arrays=True)

        # Restore scale and shift
        yq = self.y_mean + self.y_std * yq_norm

        # Shape yqs for output
        yq = _shape_for_output(yq, yq_shape)

        return yq

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute p(y|phi).

        y: (np.ndarray)
            Measurement values. Note that these are cast as integers for
            MPA regression.

        phi: (np.ndarray)
            Latent phenotype values.

        paired: (bool)
            Whether y,phi values should be treated as pairs.
            If so, y and phi must have the same number of elements.
            The shape of y will be used as output.

        Returns
        -------
            p: (np.ndarray)
                Probability of y given phi.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        # If GE, compute yhat, then p
        if self.regression_type == 'GE':

            # Compute y_hat
            yhat = self.phi_to_yhat(phi)

            # Comptue p_y_given_phi using yhat
            p = self.p_of_y_given_yhat(y, yhat, paired=True)

        # Otherwise, just compute p
        elif self.regression_type == 'MPA':

            # Cast y as integers
            y = y.astype(int)

            # Make sure all y values are valid
            check(np.all(y >= 0),
                  f"Negative values for y are invalid for MAP regression")

            check(np.all(y < self.Y),
                  f"Some y values exceed the number of bins {self.Y}")

            # Unfix phi
            phi_unfixed = self.unfixed_phi_mean + phi * self.unfixed_phi_std

            # Get values for all bins
            #p_of_all_y_given_phi = self.model.p_of_all_y_given_phi(phi_unfixed)
            p_of_all_y_given_phi = \
                self.layer_measurement_process.p_of_all_y_given_phi(
                    phi_unfixed,
                    use_arrays=True)

            # Extract y-specific elements
            _ = np.newaxis
            all_y = np.arange(self.Y).astype(int)
            y_ix = (y[:, _] == all_y[_, :])
            p = p_of_all_y_given_phi[y_ix]

        else:
            assert False, 'This should not happen.'

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    def p_of_y_given_yhat(self, y, yhat, paired=False):
        """
        Compute p(y|yhat); for GE models only.

        Parameters
        ----------
        y: (float or array-like of floats)
            Measurement values.

        yhat: (float or array-like of floats)
            Latent phenotype values.

        paired: (bool)
            Whether y,yhat values should be treated as pairs.
            If so, y and yhat must have the same number of elements.
            The shape of y will be used as output.

        Returns
        -------
            p: (float or array-like of floats)
                Probability of y given yhat.
        """
        check(self.regression_type == 'GE',
              f'Only works for GE models.')

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == yhat_shape,
                  f"y shape={y_shape} does not match yhat shape={yhat_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, yhat = _broadcast_arrays(y, yhat)

            # Set output shape
            p_shape = y_shape + yhat_shape

        # Ravel arrays
        y = y.ravel()
        yhat = yhat.ravel()

        # Normalize
        y_norm = (y - self.y_mean)/self.y_std
        yhat_norm = (yhat - self.y_mean)/self.y_std

        # Get layer
        layer = self.layer_noise_model

        # Compute p_norm using layer
        p_norm = layer.p_of_y_given_yhat(y_norm, yhat_norm, use_arrays=True)

        # Unnormalize p
        p = p_norm / self.y_std

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    def p_of_y_given_x(self, y, x):
        """
        Compute p(y|x).

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
            phi = self.x_to_phi(x)
            yhat = self.phi_to_yhat(phi)
            p = self.p_of_y_given_yhat(y, yhat)
            return p

        elif self.regression_type=='MPA':

            # check that entered y (specifying bin number) is an integer
            check(isinstance(y, int),
                  'type(y), specifying bin number, must be of type int')

            # check that entered bin number doesn't exceed max bins
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
        Save the trained MAVE-NN model.

        Note: This does not save the training data, only training data
        statistics.

        Parameters
        ----------
        filename: (str)
            filename of the saved model.

        verbose: (bool)
            Whether to provide user feedback.

        Returns
        -------
        None.
        """
        # Create config_dict
        config_dict = {
            'model_kwargs': self.arg_dict,
            'unfixed_phi_mean': self.unfixed_phi_mean,
            'unfixed_phi_std': self.unfixed_phi_std,
            'y_std': self.y_std,
            'y_mean': self.y_mean,
            'x_stats': self.x_stats,
            'y_stats': self.y_stats,
            'history': self.history,
            'info_for_layers_dict': self.info_for_layers_dict
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



