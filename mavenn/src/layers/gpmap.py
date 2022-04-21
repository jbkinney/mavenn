"""gpmap.py: Defines layers representing G-P maps."""
# Standard imports
import numpy as np
from collections.abc import Iterable
import re
from typing import Optional

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras.layers import Layer, Dense

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors
from mavenn.src.utils import x_to_stats, validate_seqs
from mavenn.src.reshape import _shape_for_output, \
    _get_shape_and_return_1d_array


class GPMapLayer(Layer):
    """
    Represents a general genotype-phenotype map.

    Specific functional forms for G-P maps should be
    represented by derived classes of this layer.
    """

    @handle_errors
    def __init__(self,
                 L,
                 alphabet,
                 theta_regularization=1e-3):
        """Construct layer instance."""

        # TODO: need to perform parameter checks here as in old model.py

        # Set sequence length
        check(L > 0,
              f'len(x[0])={L}; must be > 0')

        # Set sequence length
        self.L = L

        # Set alphabet
        self.alphabet = alphabet

        # Set alphabet length
        self.C = len(self.alphabet)

        # Set regularization contribution
        self.theta_regularization = theta_regularization

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.theta_regularization)

        # Initialize mask dict
        self.mask_dict = {}

        # Define regular expression for theta parameters
        self.theta_pattern = re.compile('^theta.*')

        # Call superclass constructor
        super().__init__()

    @handle_errors
    def get_config(self):
        """Return configuration dictionary."""
        base_config = super(Layer, self).get_config()
        return {'L': self.L,
                'C': self.C,
                'theta_regularization': self.theta_regularization,
                **base_config}

    @handle_errors
    def set_params(self, **kwargs):
        """Set values of layer parameters."""

        # Iterate over kwargs
        for k, v in kwargs.items():

            # Get current parameter object
            check(k in self.__dict__,
                  f'Keyword argument "{k}" is not the name of a parameter')
            check(bool(self.theta_pattern.match(k)),
                  f'Keyword argument "{k}" does not match a theta parameter')
            self_param = self.__dict__[k]

            # Type and shape v as needed
            v = np.array(v).astype(np.float).reshape(self_param.shape)

            # Mask meaningless values with zeros
            no_mask = np.full(v.shape, True, dtype=bool)
            mask = self.mask_dict.get(k, no_mask)
            v[~mask] = 0.0

            # Assign to self_param values
            self_param.assign(v)

    @handle_errors
    def get_params(self,
                   squeeze=True,
                   pop=True,
                   mask_with_nans=True):

        # Get theta_dict
        theta_dict = {k: v for (k, v) in self.__dict__.items()
                      if self.theta_pattern.match(k)
                      and isinstance(v, tf.Variable)}

        # Modify dict values as requested
        for k, v in theta_dict.items():

            # Convert to numpy array
            v = v.numpy()

            # Mask meaningless values with nans
            if mask_with_nans:
                no_mask = np.full(v.shape, True, dtype=bool)
                mask = self.mask_dict.get(k, no_mask)
                v[~mask] = np.nan

            # Squeeze out singleton dimensions
            # Pop out values form singleton arrays
            if squeeze:
                v = v.squeeze()

            if pop and v.size == 1:
                v = v.item()

            # Save modified value
            theta_dict[k] = v

        return theta_dict

    ### The following methods must be fully overridden ###

    @handle_errors
    def build(self, input_shape):
        # Call superclass build
        super().build(input_shape)

    @handle_errors
    def call(self, inputs):
        """Process layer input and return output."""
        assert False

    @handle_errors
    def x_to_phi(self, x, batch_size: Optional[int] = 64):
        """GP map to x_to_phi abstract method.

        Parameters
        ----------
        x: (str)
            Sequences.
        batch_size: Optional (int)
            Default values is 64.

        Returns
        ----------
        phi: (numpy array dtype=float32)
            Latent phenotype values.

        Note
        ----------
        This function caused the memory overflow:
        If the size of the x is large and higher order GP maps are used
        the matrix multiplication needs huge memory.
        Solution: Used the batched version of self.call
        """

        # Shape x for processing
        x, x_shape = _get_shape_and_return_1d_array(x)

        # Check seqs
        x = validate_seqs(x, alphabet=self.alphabet)
        check(len(x[0]) == self.L,
              f'len(x[0])={len(x[0])}; should be L={self.L}')

        # Encode sequences as features
        stats = x_to_stats(x=x, alphabet=self.alphabet)

        # Convert the x_ohe to tensorflow dataset with batch
        x_ohe = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(stats.pop('x_ohe'), dtype=tf.float32)).batch(batch_size)

        # Note: this is currently not diffeomorphic mode fixed.
        # Apply x_to_phi calls on batches
        phi = x_ohe.map(lambda z: self.call(z))

        # Unbatch and gather all the phi values in numpy array
        phi = np.array(list(phi.unbatch().as_numpy_iterator()))

        # Shape phi for output
        phi = _shape_for_output(phi, x_shape)

        return phi


class AdditiveGPMapLayer(GPMapLayer):
    """Represents an additive G-P map."""

    @ handle_errors
    def __init__(self, *args, **kwargs):
        """Construct layer instance."""

        # Call superclass constructor
        super().__init__(*args, **kwargs)

        """Build layer."""
        # Define theta_0
        self.theta_0 = self.add_weight(name='theta_0',
                                       shape=(1,),
                                       initializer=Constant(0.),
                                       trainable=True,
                                       regularizer=self.regularizer)

        # Define theta_lc parameters
        theta_lc_shape = (1, self.L, self.C)
        # 2202.02.04: this initializer creates problems, need to remove everywhere
        # theta_lc_init = np.random.randn(*theta_lc_shape)/np.sqrt(self.L)
        self.theta_lc = self.add_weight(name='theta_lc',
                                        shape=theta_lc_shape,
                                        # initializer=Constant(theta_lc_init),
                                        trainable=True,
                                        regularizer=self.regularizer)

    def call(self, x_lc):
        """Process layer input and return output."""
        # Shape input
        x_lc = tf.reshape(x_lc, [-1, self.L, self.C])

        phi = self.theta_0 + \
            tf.reshape(K.sum(self.theta_lc * x_lc, axis=[1, 2]),
                       shape=[-1, 1])

        return phi


class PairwiseGPMapLayer(GPMapLayer):
    """Represents a pairwise G-P map."""

    @ handle_errors
    def __init__(self, mask_type, *args, **kwargs):
        """Construct layer instance."""

        # Call superclass constructor
        super().__init__(*args, **kwargs)

        # Define theta_0
        self.theta_0 = self.add_weight(name='theta_0',
                                       shape=(1,),
                                       initializer=Constant(0.),
                                       trainable=True,
                                       regularizer=self.regularizer)

        # Define theta_lc parameters
        theta_lc_shape = (1, self.L, self.C)
        theta_lc_init = np.random.randn(*theta_lc_shape) / np.sqrt(self.L)
        self.theta_lc = self.add_weight(name='theta_lc',
                                        shape=theta_lc_shape,
                                        initializer=Constant(theta_lc_init),
                                        trainable=True,
                                        regularizer=self.regularizer)

        # Set mask type
        self.mask_type = mask_type
        check(self.mask_type in ['neighbor', 'pairwise'],
              f'self.mask_type={repr(self.mask_type)}; must be'
              f'one of ["neighbor","pairwise"]')

        # Create mask for theta_lclc
        ls = np.arange(self.L).astype(int)
        ls1 = np.tile(ls.reshape([1, self.L, 1, 1, 1]),
                      [1, 1, self.C, self.L, self.C])
        ls2 = np.tile(ls.reshape([1, 1, 1, self.L, 1]),
                      [1, self.L, self.C, 1, self.C])
        if self.mask_type == 'pairwise':
            mask = (ls2 - ls1 >= 1)
        elif self.mask_type == 'neighbor':
            mask = (ls2 - ls1 == 1)
        else:
            assert False, "This should not happen."
        self.mask_dict['theta_lclc'] = mask

        # Define theta_lclc parameters
        theta_lclc_shape = (1, self.L, self.C, self.L, self.C)
        theta_lclc_init = np.random.randn(
            *theta_lclc_shape) / np.sqrt(self.L**2)
        theta_lclc_init *= self.mask_dict['theta_lclc']
        self.theta_lclc = self.add_weight(name='theta_lclc',
                                          shape=theta_lclc_shape,
                                          initializer=Constant(
                                              theta_lclc_init),
                                          trainable=True,
                                          regularizer=self.regularizer)

    def call(self, x_lc):
        """Process layer input and return output."""

        # Compute phi
        phi = self.theta_0
        phi = phi + tf.reshape(K.sum(self.theta_lc *
                                     tf.reshape(x_lc, [-1, self.L, self.C]),
                                     axis=[1, 2]),
                               shape=[-1, 1])
        phi = phi + tf.reshape(K.sum(self.theta_lclc *
                                     self.mask_dict['theta_lclc'] *
                                     tf.reshape(x_lc,
                                                [-1, self.L, self.C, 1, 1]) *
                                     tf.reshape(x_lc,
                                                [-1, 1, 1, self.L, self.C]),
                                     axis=[1, 2, 3, 4]),
                               shape=[-1, 1])

        return phi

    @ handle_errors
    def get_config(self):
        """Return configuration dictionary."""

        # Get base config of superclass
        base_config = super().get_config()

        # Add new param from __init__() to dict and return
        return {'mask_type': self.mask_type,
                **base_config}


class MultilayerPerceptronGPMap(GPMapLayer):
    """Represents an MLP G-P map."""

    @ handle_errors
    def __init__(self,
                 *args,
                 hidden_layer_sizes=(10, 10, 10),
                 hidden_layer_activation='relu',
                 features='additive',
                 **kwargs):

        # Check and set hidden layer sizes
        check(isinstance(hidden_layer_sizes, Iterable),
              f'type(hidden_layer_sizes)={type(hidden_layer_sizes)}; '
              f'must be Iterable.')
        check(all([x >= 1 for x in hidden_layer_sizes]),
              f'all elements of hidden_layer_sizes={hidden_layer_sizes}'
              f'must be >= 1')
        check(all([isinstance(x, int) for x in hidden_layer_sizes]),
              f'all elements of hidden_layer_sizes={hidden_layer_sizes}'
              f'must be int.')
        self.hidden_layer_sizes = hidden_layer_sizes

        # Check and set features
        allowed_features = ['additive', 'neighbor', 'pairwise']
        check(features in allowed_features,
              f'features={repr(features)}; must be one of {allowed_features}.')
        self.features = features

        # Initialize array to hold layers
        self.layers = []

        # Set activation
        self.hidden_layer_activation = hidden_layer_activation
        super().__init__(*args, **kwargs)

    @ handle_errors
    def build(self, input_shape):

        # Determine input shape
        L = self.L
        C = self.C
        if self.features == 'additive':
            self.num_features = L * C
        elif self.features == 'neighbor':
            self.num_features = L * C + (L - 1) * (C**2)
        elif self.features == 'pairwise':
            self.num_features = L * C + L * (L - 1) * (C**2) / 2
        self.x_shape = (input_shape[0], int(self.num_features))

        # Create mask
        ls = np.arange(self.L).astype(int)
        ls1 = np.tile(ls.reshape([L, 1, 1, 1]),
                      [1, C, L, C])
        ls2 = np.tile(ls.reshape([1, 1, L, 1]),
                      [L, C, 1, C])
        if self.features in ['neighbor', 'pairwise']:
            if self.features == 'pairwise':
                mask_lclc = (ls2 - ls1 >= 1)
            else:
                mask_lclc = (ls2 - ls1 == 1)
            mask_vec = np.reshape(mask_lclc, L * C * L * C)
            self.mask_ints = np.arange(L * C * L * C, dtype=int)[mask_vec]
        elif self.features == 'additive':
            self.mask_ints = None
        else:
            assert False, "This should not work"

        # Make sure self.layers is empty
        self.layers = []

        if len(self.hidden_layer_sizes) >= 1:
            # Add hidden layer #1
            size = self.hidden_layer_sizes[0]
            self.layers.append(
                Dense(units=size,
                      activation=self.hidden_layer_activation,
                      input_shape=self.x_shape,
                      kernel_regularizer=self.regularizer,
                      bias_regularizer=self.regularizer)
            )

            # Add rest of hidden layers
            for size in self.hidden_layer_sizes[1:]:
                self.layers.append(
                    Dense(units=size,
                          activation=self.hidden_layer_activation,
                          kernel_regularizer=self.regularizer,
                          bias_regularizer=self.regularizer)
                )

            # Add output layer
            self.layers.append(
                Dense(units=1,
                      activation='linear',
                      kernel_regularizer=self.regularizer,
                      bias_regularizer=self.regularizer)
            )
        elif len(self.hidden_layer_sizes) == 0:
            # Add single layer; no hidden nodes
            self.layers.append(
                Dense(units=1,
                      activation='linear',
                      input_shape=self.x_shape,
                      kernel_regularizer=self.regularizer,
                      bias_regularizer=self.regularizer)
            )
        else:
            assert False, 'This should not happen.'

        # Build superclass
        super().build(input_shape)

    def call(self, x_add):
        """Process layer input and return output."""

        # Create input features
        if self.features == 'additive':
            tensor = x_add
        elif self.features in ['neighbor', 'pairwise']:
            L = self.L
            C = self.C
            x___lc = tf.reshape(x_add, [-1, 1, 1, L, C])
            x_lc__ = tf.reshape(x_add, [-1, L, C, 1, 1])
            x_lclc = x___lc * x_lc__
            x_pair = tf.reshape(x_lclc, [-1, L * C * L * C])

            # Only use relevant columns
            x_2pt = tf.gather(x_pair, self.mask_ints, axis=1)

            # Make input tensor
            tensor = tf.concat([x_add, x_2pt], axis=1)

        # Run tensor through layers
        for layer in self.layers:
            tensor = layer(tensor)
        phi = tensor

        return phi

    @ handle_errors
    def set_params(self, theta_0=None, theta_lc=None):
        """
        Does nothing for MultilayerPerceptronGPMap
        """
        print('Warning: MultilayerPerceptronGPMap.set_params() does nothing.')

    @ handle_errors
    def get_params(self):
        """
        Get values of layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        param_dict: (dict)
            Dictionary containing model parameters.
        """

        #  Fill param_dict
        param_dict = {}
        param_dict['theta_mlp'] = [layer.get_weights()
                                   for layer in self.layers]

        return param_dict


class KOrderGPMap(GPMapLayer):
    """Represents a arbitrary interaction G-P map."""

    @ handle_errors
    def __init__(self,
                 interaction_order: Optional[int] = 0,
                 *args, **kwargs):
        """Construct layer instance."""

        # Call superclass constructor
        super().__init__(*args, **kwargs)

        """Build layer."""
        # Set interaction order
        self.interaction_order = interaction_order
        # Check that interaction order is less than the sequence length
        check(self.interaction_order <= self.L,
              f'self.interaction_order {self.interaction_order} must be'
              f' equal to or less than sequence length {self.L}')

        # Initialize the theta dictionary
        self.theta_dict = {}

        # Create the theta_0 name.
        theta_0_name = f'theta_0'
        # Define theta_0 weight
        self.theta_dict[theta_0_name] = self.add_weight(name=theta_0_name,
                                                        shape=(1,),
                                                        initializer=Constant(
                                                            0.),
                                                        trainable=True,
                                                        regularizer=self.regularizer)

        # Create the theta_lc
        theta_shape = (1,)
        seq_len_arange = np.arange(self.L).astype(int)
        # Create masking dictionary
        self.mask_dict = {}
        # Loop over interaction order
        for k in range(interaction_order):
            theta_name = f'theta_{k+1}'
            # Add L, C to the shape of theta_k for each interaction order k
            theta_shape = theta_shape + (self.L, self.C)
            self.theta_dict[theta_name] = self.add_weight(name=theta_name,
                                                          shape=theta_shape,
                                                          initializer=RandomNormal(),
                                                          trainable=True,
                                                          regularizer=self.regularizer)
            # Create list of indices
            # which should be False or True based on level of interactions.
            # The ls_dict is analogous to l, l', l'', ... in MAVE-NN paper
            ls_dict = {}
            # starting location of L,C characters in the shape lists
            lc_loc = 1
            for w in range(k + 1):
                ls_part_shape = [1] * len(theta_shape)
                ls_tile_shape = list(theta_shape)
                ls_part_shape[lc_loc] = self.L
                ls_tile_shape[lc_loc] = 1
                ls = np.tile(seq_len_arange.reshape(
                    ls_part_shape), ls_tile_shape)
                ls_dict[f'ls_{w}'] = ls
                lc_loc = lc_loc + 2
            m_dict = {}
            for w in range(k):
                m_dict[f'm_{w}'] = ls_dict[f'ls_{w+1}'] > ls_dict[f'ls_{w}']

            mask = True
            for key in m_dict.keys():
                mask = m_dict[key] * mask
            # Create mask array
            self.mask_dict[theta_name] = mask

    def call(self, x_lc):
        """Process layer input and return output."""

        # Get the interaction order
        interaction_order = self.interaction_order

        # 0-th order interaction
        theta_0_name = f'theta_0'
        phi = self.theta_dict[theta_0_name]

        # Loop over interaction order
        theta_shape = (1,)

        for k in range(interaction_order):
            theta_name = f'theta_{k+1}'
            theta_shape = theta_shape + (self.L, self.C)
            # Find the axis shape (order) which we should sum the array
            axis_shape = np.arange(1, len(theta_shape))
            # Location of L, C char in x_lc reshape arguments
            lc_loc = 1
            # To find interactions we need to find x_lc*x_l'c'*...
            # Here we call that multiplication x_mult
            x_mult = 1
            for w in range(k + 1):
                # Find the shape of x_lc
                x_shape_k_order = [1] * len(theta_shape)
                x_shape_k_order[0] = -1
                x_shape_k_order[lc_loc] = self.L
                x_shape_k_order[lc_loc + 1] = self.C
                x_mult = x_mult * tf.reshape(x_lc, x_shape_k_order)
                lc_loc = lc_loc + 2

            phi = phi + \
                tf.reshape(
                    K.sum(self.theta_dict[theta_name] * self.mask_dict[theta_name] * x_mult, axis=axis_shape), [-1, 1])

        return phi
