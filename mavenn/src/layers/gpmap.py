"""gpmap.py: Defines layers representing G-P maps."""
# Standard imports
import numpy as np

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors

class GPMapLayer(Layer):
    """
    Represents a general genotype-phenotype map.

    Specific functional forms for G-P maps should be
    represented by derived classes of this layer.
    """

    @handle_errors
    def __init__(self,
                 L,
                 C,
                 theta_regularization,
                 mask_type=None):
        """Construct layer instance."""
        # Set sequence length
        self.L = L

        # Set alphabet length
        self.C = C

        # Set regularization contribution
        self.theta_regularization = theta_regularization

        # Set mask type
        self.mask_type = mask_type

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
    def build(self, input_shape):
        """Build layer."""
        super().build(input_shape)

    ### The following methods must be fully overridden

    def call(self, inputs):
        """Process layer input and return output."""
        assert False
        return np.nan

    def set_params(self, **kwargs):
        """Set values of layer parameters."""
        assert False

    def get_params(self):
        """Get values of layer parameters."""
        assert False
        return {}


class AdditiveGPMapLayer(GPMapLayer):
    """Represents an additive G-P map."""

    @handle_errors
    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

    @handle_errors
    def build(self, input_shape):
        """Build layer."""
        # Define theta_0
        self.theta_0 = self.add_weight(name='theta_0',
                                       shape=(1,),
                                       initializer=Constant(0.),
                                       trainable=True)

        # Define theta_lc parameters
        theta_lc_shape = (1, self.L, self.C)
        theta_lc_init = np.random.randn(*theta_lc_shape)/np.sqrt(self.L)
        self.theta_lc = self.add_weight(name='theta_lc',
                                        shape=theta_lc_shape,
                                        initializer=Constant(theta_lc_init),
                                        trainable=True)
        # Call superclass build
        super().build(input_shape)

    def call(self, x_lc):
        """Process layer input and return output."""
        # Shape input
        x_lc = tf.reshape(x_lc, [-1, self.L, self.C])

        phi = self.theta_0 + \
              tf.reshape(K.sum(self.theta_lc * x_lc, axis=[1, 2]),
                         shape=[-1, 1])

        # Compute regularization loss
        norm_sq = tf.norm(self.theta_0) ** 2 + tf.norm(self.theta_lc) ** 2
        self.add_loss(self.theta_regularization * norm_sq)

        return phi

    @handle_errors
    def set_params(self, theta_0=None, theta_lc=None):
        """
        Set values of layer parameters.

        Parameters
        ----------
        theta_0: (float)

        theta_lc: (np.ndarray)
            Shape (L,C)

        Returns
        -------
        None
        """
        # Check theta_0
        if theta_0 is not None:
            check(isinstance(theta_0, float),
                  f'type(theta_0)={theta_0}; must be float')

        # Check theta_lc
        if theta_lc is not None:
            check(isinstance(theta_lc, np.ndarray),
                  f'type(theta_lc)={theta_lc}; must be np.ndarray')
            check(theta_lc.size == self.L * self.C,
                   f'theta_lc.size={repr(theta_lc.size)}; '
                   f'must be ({self.L * self.C}).')
            theta_lc = theta_lc.reshape([1, self.L, self.C])

        # Set weight values
        self.set_weights([np.array([theta_0]), theta_lc])

    @handle_errors
    def get_params(self):
        """
        Get values of layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        param_dict: (dict)
            Dictionary containing model parameters. Model parameters are
            returned as matrices, NOT as individual named parameters, and are
            NOT gauge-fixed.
        """
        # Get list of weights
        param_list = self.get_weights()

        #  Fill param_dict
        param_dict = {}
        param_dict['theta_0'] = param_list[0]
        param_dict['theta_lc'] = param_list[1].reshape([self.L, self.C])

        return param_dict


class PairwiseGPMapLayer(GPMapLayer):
    """Represents a pairwise G-P map."""

    @handle_errors
    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

        # Set mask type
        check(self.mask_type in ['neighbor', 'pairwise'],
              f'self.mask_type={repr(self.mask_type)}; must be'
              f'one of ["neighbor","pairwise"]')

        # Create mask
        ls = np.arange(self.L).astype(int)
        ls1 = np.tile(ls.reshape([1, self.L, 1, 1, 1]),
                      [1, 1, self.C, self.L, self.C])
        ls2 = np.tile(ls.reshape([1, 1, 1, self.L, 1]),
                      [1, self.L, self.C, 1, self.C])
        if self.mask_type == 'pairwise':
            self.mask = (ls2 - ls1 >= 1)
        elif self.mask_type == 'neighbor':
            self.mask = (ls2 - ls1 == 1)
        else:
            assert False, "This should not work"

    @handle_errors
    def get_config(self):
        """Return configuration dictionary."""
        base_config = super().get_config()
        return {'mask_type': self.mask_type,
                **base_config}

    @handle_errors
    def build(self, input_shape):
        """Build layer."""
        # Define theta_0
        self.theta_0 = self.add_weight(name='theta_0',
                                       shape=(1,),
                                       initializer=Constant(0.),
                                       trainable=True)

        # Define theta_lc parameters
        theta_lc_shape = (1, self.L, self.C)
        theta_lc_init = np.random.randn(*theta_lc_shape)/np.sqrt(self.L)
        self.theta_lc = self.add_weight(name='theta_lc',
                                        shape=theta_lc_shape,
                                        initializer=Constant(theta_lc_init),
                                        trainable=True)

        # Define theta_lclc parameters
        theta_lclc_shape = (1, self.L, self.C, self.L, self.C)
        theta_lclc_init = np.random.randn(*theta_lclc_shape)/np.sqrt(self.L**2)
        theta_lclc_init *= self.mask
        self.theta_lclc = self.add_weight(name='theta_lclc',
                                          shape=theta_lclc_shape,
                                          initializer=Constant(theta_lclc_init),
                                          trainable=True)

        # Call superclass build
        super().build(input_shape)

    def call(self, x_lc):
        """Process layer input and return output."""
        # Compute phi
        phi = self.theta_0
        phi = phi + tf.reshape(K.sum(self.theta_lc *
                                     tf.reshape(x_lc, [-1, self.L, self.C]),
                                     axis=[1, 2]),
                               shape=[-1, 1])
        phi = phi + tf.reshape(K.sum(self.theta_lclc *
                                     self.mask *
                                     tf.reshape(x_lc,
                                         [-1, self.L, self.C, 1, 1]) *
                                     tf.reshape(x_lc,
                                         [-1, 1, 1, self.L, self.C]),
                                     axis=[1, 2, 3, 4]),
                               shape=[-1, 1])

        # Compute regularization loss
        norm_sq = tf.norm(self.theta_0) ** 2 + \
                  tf.norm(self.theta_lc) ** 2 + \
                  tf.norm(self.theta_lclc) ** 2
        self.add_loss(self.theta_regularization * norm_sq)

        return phi

    @handle_errors
    def set_params(self, theta_0=None, theta_lc=None, theta_lclc=None):
        """
        Set values of layer parameters.

        Parameters
        ----------
        theta_0: (float)

        theta_lc: (np.ndarray)
            Shape (L,C)

        theta_lclc: (np.ndarray)
            Shape (L,C,L,C)

        Returns
        -------
        None
        """
        # Check theta_0
        if theta_0 is not None:
            check(isinstance(theta_0, float),
                  f'type(theta_0)={theta_0}; must be float')

        # Check theta_lc
        if theta_lc is not None:
            check(isinstance(theta_lc, np.ndarray),
                  f'type(theta_lc)={theta_lc}; must be np.ndarray')
            check(theta_lc.size == self.L * self.C,
                   f'theta_lc.size={repr(theta_lc.size)}; '
                   f'must be ({self.L * self.C}).')
            theta_lc = theta_lc.reshape([1, self.L, self.C])

        # Check theta_lclc
        if theta_lclc is not None:
            check(isinstance(theta_lclc, np.ndarray),
                  f'type(theta_lclc)={theta_lclc}; must be np.ndarray')
            check(theta_lclc.size == self.L * self.C * self.L * self.C,
                   f'theta_lclc.size={repr(theta_lclc.size)}; '
                   f'must be ({self.L * self.C * self.L * self.C}).')
            theta_lclc = theta_lclc.reshape([1, self.L, self.C, self.L, self.C])

        # Set weight values
        self.set_weights([np.array([theta_0]), theta_lc, theta_lclc])

    @handle_errors
    def get_params(self):
        """
        Get values of layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        param_dict: (dict)
            Dictionary containing model parameters. Model parameters are
            returned as matrices, NOT as individual named parameters, and are
            NOT gauge-fixed.
        """
        # Get list of weights
        param_list = self.get_weights()

        #  Fill param_dict
        param_dict = {}
        param_dict['theta_0'] = param_list[0]
        param_dict['theta_lc'] = param_list[1].reshape([self.L, self.C])
        masked_theta_lclc = param_list[2]
        masked_theta_lclc[~self.mask] = np.nan
        param_dict['theta_lclc'] = \
            masked_theta_lclc.reshape([self.L, self.C, self.L, self.C])

        return param_dict
