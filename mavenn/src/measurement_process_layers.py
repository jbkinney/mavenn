import numpy as np
import pandas as pd
import pdb

# Used in wrapper
from functools import wraps

# Scipy imports
from scipy.special import betaincinv, erfinv
from scipy.stats import expon

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.initializers import Constant
from tensorflow.math import tanh, sigmoid
from tensorflow.keras.layers import Layer

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors

eps = 1E-10
pi = np.pi
e = np.exp(1)

# To clarify equations
Log = K.log
LogGamma = tf.math.lgamma
Exp = K.exp
Sqrt = K.sqrt
Square = K.square


def handle_arrays(func):
    """
    Decorator function that allows functions that take and return
    tensors to operate on numpy arrays. If use_arrays=True is set
    as a keyword argument, every input of one of the input_types
    is converted to a tf.constant before being passed to func. Then,
    all outputs from func that are tf.Tensor objects are converted to
    np.ndarrays.
    """

    # Input types to convert to tf.Tensor objects upon input
    input_types = (np.ndarray, list, pd.Series)

    # Use @wraps to preserve docstring
    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Get use_arrays keyword argument.
        use_arrays = kwargs.pop('use_arrays', False)

        # Make sure use_arrays is boolean
        check(isinstance(use_arrays, bool),
              f'use_arrays={use_arrays}; must be bool.')

        # If use_arrays is selected
        if use_arrays:

            # Convert positional arguments
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, input_types):
                    args[i] = tf.constant(arg, dtype=tf.float32)

            # Convert keyword arguments
            for key, arg in kwargs.items():
                if isinstance(arg, input_types):
                    kwargs[key] = tf.constant(arg, dtype=tf.float32)

            # Pass inputs to func and get outputs
            result = func(*args, **kwargs)

            # Convert results within a tuple
            if isinstance(result, tuple):
                result = (r.numpy() if isinstance(r, tf.Tensor) else r
                          for r in result)

            # Convert a sole result
            else:
                result = result.numpy() if isinstance(result, tf.Tensor) \
                            else result

        # Otherwise, just use raw function.
        else:
            result = func(*args, **kwargs)

        # Retun (potentially) adjusted results
        return result

    # Return wrapped function
    return wrapped_func


class MPAMeasurementProcessLayer(Layer):
    def __init__(self,
                 info_for_layers_dict,
                 number_bins,
                 **kwargs):
        self.number_bins = number_bins
        self.info_for_layers_dict = info_for_layers_dict
        super().__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """
        parameters
        ----------

        inputs: (tf.Tensor)
            A (B,2*Y) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:, 0:Y] is p(y|phi)
            inputs[:, Y:2Y] is c_my

        returns
        -------
            A (B,) tensor containing negative log likelihood contributions
        """

        # this is yhat
        p_y_given_phi = inputs[:, 0:self.number_bins]

        # these are the labels
        c_my = inputs[:, self.number_bins:]

        negative_log_likelihood = -K.sum(c_my * K.log(p_y_given_phi), axis=1)
        c_m = K.sum(c_my, axis=1)

        # Add I_like metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
                        K.sum(negative_log_likelihood) / K.sum(c_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_like", aggregation="mean")

        return negative_log_likelihood


class AffineLayer(Layer):
    """
    Represents an affine map from phi to yhat.
    """

    @handle_errors
    def __init__(self,
                 eta_regularization,
                 monotonic,
                 **kwargs):

        # Whether to make monotonic function
        self.monotonic = monotonic

        # Create function that returns a kernel constraint
        # based on self.monotonic
        self.constraint = lambda: non_neg() if self.monotonic else None

        # Set regularizer
        self.eta = eta_regularization
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        # Call superclass constructor
        super(GlobalEpistasisLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a',
                                 shape=(1,),
                                 initializer=Constant(0.),
                                 trainable=True,
                                 regularizer=self.regularizer)

        self.b = self.add_weight(name='b',
                                   shape=(1,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    # Just an alias for phi_to_yhat, for tensors only
    def call(self, phi):
        return self.phi_to_yhat(phi)

    # yhat as function of phi
    @handle_arrays
    def phi_to_yhat(self, phi):
        yhat = self.a + self.b * phi
        return yhat


class GlobalEpistasisLayer(Layer):
    """
    Represents a global epistasis layer.
    """

    @handle_errors
    def __init__(self,
                 K,
                 eta_regularization,
                 monotonic,
                 **kwargs):

        # Whether to make monotonic function
        self.monotonic = monotonic

        # Create function that returns a kernel constraint
        # based on self.monotonic
        self.constraint = lambda: non_neg() if self.monotonic else None

        # Set number of hidden nodes
        self.K = K

        # Set regularizer
        self.eta = eta_regularization
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        # Call superclass constructor
        super(GlobalEpistasisLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.a_0 = self.add_weight(name='a_0',
                                   shape=(1,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        b_k_dist = expon(scale=1/self.K)
        self.b_k = self.add_weight(name='b_k',
                                   shape=(self.K,),
                                   initializer=Constant(b_k_dist.rvs(self.K)),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.c_k = self.add_weight(name='c_k',
                                   shape=(self.K,),
                                   initializer=Constant(1.),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    # Just an alias for phi_to_yhat, for tensors only
    def call(self, phi):
        return self.phi_to_yhat(phi)

    # yhat as function of phi
    @handle_arrays
    def phi_to_yhat(self, phi):
        yhat = self.a_0 + tf.reshape(
                    K.sum(self.b_k * tanh(self.c_k * phi + self.d_k), axis=1),
                    shape=[-1, 1])
        return yhat



class NoiseModelLayer(Layer):
    """
    Custom Keras Layer representing a GE noise model.
    Specific noise model layers inherit from this class.
    """

    def __init__(self,
                 info_for_layers_dict,
                 polynomial_order=2,
                 eta_regularization=0.01,
                 **kwargs):

        # order of polynomial which defines log_sigma's dependence on y_hat
        self.K = polynomial_order
        self.eta = eta_regularization
        self.info_for_layers_dict = info_for_layers_dict
        self.regularizer = tf.keras.regularizers.L2(self.eta)
        super().__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'info_for_layers_dict': self.info_for_layers_dict,
                'polynomial_order': self.K,
                'eta_regularization': self.eta}

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """
        parameters
        ----------

        inputs: (tf.Tensor)
            A (B,2) tensor containing both yhat and ytrue, where B is batch
            size.

        returns
        -------
            A (B,) tensor containing negative log likelihood contributions
        """

        # this is yhat
        yhat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        # Compute negative log likelihood
        nlls = self.compute_nlls(yhat=yhat,
                                 ytrue=ytrue)

        # Compute I_like metric from nlls
        H_y = self.info_for_layers_dict.get('H_y_norm', np.nan)
        H_y_given_phi = K.mean(np.log2(e) * nlls)
        I_like = H_y - H_y_given_phi
        self.add_metric(I_like, name="I_like", aggregation="mean")

        return nlls

    @handle_arrays
    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """
        Compute p(y|yhat).
        """

        # Compute negative log likeliihood
        nll_arr = self.compute_nlls(yhat=yhat,
                                    ytrue=y)

        # Convert to p_y_given_yhat and return
        p_y_given_yhat = K.exp(-nll_arr)

        return p_y_given_yhat

    @handle_arrays
    def sample_y_given_yhat(self, yhat):
        # Draw random quantiles
        q = tf.constant(np.random.rand(*yhat.shape), dtype=tf.float32)

        # Compute corresponding quantiles and return
        y = self.yhat_to_yq(yhat, q)
        return y

    ### The following functions must be overridden
    def compute_params(self, yhat, ytrue):
        assert False, 'Function must be overridden'

    def compute_nlls(self, yhat, ytrue):
        assert False, 'Function must be overridden'

    def yhat_to_yq(self, yhat, q):
        assert False, 'Function must be overridden'

    def yhat_to_ymean(self, yhat):
        assert False, 'Function must be overridden'

    def yhat_to_ystd(self, yhat):
        assert False, 'Function must be overridden'


class GaussianNoiseModelLayer(NoiseModelLayer):
    """
    Custom kears Layer representing a
    Gaussian noise model for GE regression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a',
                                 shape=(self.K + 1, 1),
                                 initializer="TruncatedNormal",
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    @handle_arrays
    def compute_params(self, yhat):
        # Have to treat 0'th order term separately because of NaN bug.
        logsigma = self.a[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            logsigma += self.a[k] * K.pow(yhat, k)

        return logsigma

    @handle_arrays
    def compute_nlls(self, yhat, ytrue, use_arrays=False):
        # Compute logsigma and sigma
        logsigma = self.compute_params(yhat)
        sigma = K.exp(logsigma)

        # Compute nlls
        nlls = \
            0.5 * K.square((ytrue - yhat) / sigma) \
            + logsigma \
            + 0.5*np.log(2*pi)

        return nlls

    @handle_arrays
    def yhat_to_yq(self, yhat, q):
        sigma = K.exp(self.compute_params(yhat))
        yq = yhat + sigma * np.sqrt(2) * erfinv(2 * q.numpy() - 1)
        return yq

    @handle_arrays
    def yhat_to_ymean(self, yhat):
        return yhat

    @handle_arrays
    def yhat_to_ystd(self, yhat):
        sigma = K.exp(self.compute_params(yhat))
        return sigma


class CauchyNoiseModelLayer(NoiseModelLayer):
    """
    Custom keras Layer representing a
    Cauchy noise model for GE regression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a',
                                 shape=(self.K + 1, 1),
                                 initializer="TruncatedNormal",
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    ### Overloading functions
    @handle_arrays
    def compute_params(self, yhat):
        # Have to treat 0'th order term separately because of NaN bug.
        loggamma = self.a[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            loggamma += self.a[k] * K.pow(yhat, k)

        return loggamma

    @handle_arrays
    def compute_nlls(self, yhat, ytrue):

        # Compute loggamma
        loggamma = self.compute_params(yhat)

        # Compute nlls
        nlls = \
            K.log(K.exp(2*loggamma) + K.square(ytrue - yhat) + eps) \
            - loggamma \
            + np.log(pi)

        return nlls

    @handle_arrays
    def yhat_to_yq(self, yhat, q):
        gamma = K.exp(self.compute_params(yhat))
        yq = yhat + gamma * tf.math.tan(np.pi * (q - 0.5))
        return yq

    @handle_arrays
    def yhat_to_ymean(self, yhat):
        return yhat

    @handle_arrays
    def yhat_to_ystd(self, yhat):
        sigma = K.exp(self.compute_params(yhat))
        return sigma


class SkewedTNoiseModelLayer(NoiseModelLayer):
    """
    Custom keras Layer representing a
    SkewedT noise model for GE regression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w_a = self.add_weight(name='w_a',
                                   shape=(self.K+1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.w_b = self.add_weight(name='w_b',
                                   shape=(self.K+1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.w_s = self.add_weight(name='w_s',
                                   shape=(self.K+1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "w_a": self.w_a,
                "w_b": self.w_b, "w_s": self.w_s}

    # Compute the mode as a function of a and b
    @handle_arrays
    def t_mode(self, a, b):
        t_mode = (a - b) * K.sqrt(a + b) \
                 / (K.sqrt(2 * a + 1) * K.sqrt(2 * b + 1))
        return t_mode

    # Compute mean
    @handle_arrays
    def t_mean(self, a, b):
        return 0.5 * (a - b) * np.sqrt(a + b) * np.exp(
            tf.math.lgamma(a - 0.5)
            + tf.math.lgamma(b - 0.5)
            - tf.math.lgamma(a)
            - tf.math.lgamma(b)
            )

    @handle_arrays
    def t_std(self, a, b):
        t_expected = self.t_mean(a, b)
        tsq_expected = 0.25 * (a + b) * \
                       ((a - b) ** 2 + (a - 1) + (b - 1)) / ((a - 1) * (b - 1))
        return K.sqrt(tsq_expected - t_expected ** 2)


    @handle_arrays
    def t_quantile(self, q, a, b):
        x_q = tf.constant(betaincinv(a.numpy(), b.numpy(), q.numpy()),
                          dtype=tf.float32)
        t_q = (2 * x_q - 1) * K.sqrt(a + b) / K.sqrt(1 - (2 * x_q - 1) ** 2)
        return t_q

    ### Overloading functions
    @handle_arrays
    def compute_params(self, yhat):
        # Have to treat 0'th order term separately because of NaN bug.
        log_a = self.w_a[0]
        log_b = self.w_b[0]
        log_s = self.w_s[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            log_a += self.w_a[k] * K.pow(yhat, k)
            log_b += self.w_b[k] * K.pow(yhat, k)
            log_s += self.w_s[k] * K.pow(yhat, k)

        # Compute a, b, s in terms of trainable parameters
        a = K.exp(log_a)
        b = K.exp(log_b)
        s = K.exp(log_s)

        return a, b, s

    @handle_arrays
    def compute_nlls(self, yhat, ytrue):
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute t_mode
        t_mode = self.t_mode(a, b)

        # Compute t values
        t = t_mode + (ytrue - yhat)/s

        # Compute useful argument
        arg = t / K.sqrt(a + b + K.square(t))

        # Compute negative log likelihood contributions
        nlls = -(
            (a + 0.5) * K.log(1 + arg) +
            (b + 0.5) * K.log(1 - arg) +
            -(a + b - 1) * K.log(2.0) +
            -0.5 * K.log(a + b) +
            tf.math.lgamma(a + b) +
            -tf.math.lgamma(a) +
            -tf.math.lgamma(b) +
            -K.log(s)
            )

        return nlls

    @handle_arrays
    def yhat_to_yq(self, yhat, q):
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute t_mode
        t_mode = self.t_mode(a, b)

        # Compute random t values
        t_q = self.t_quantile(q, a, b)

        # Compute and return y
        yq = (t_q - t_mode) * s + yhat
        return yq

    @handle_arrays
    def yhat_to_ymean(self, yhat):
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute mean and mode of t distribution
        t_mean = self.t_mean(a, b)
        t_mode = self.t_mode(a, b)

        # Compute ymean
        ymean = s * (t_mean - t_mode) + yhat

        return ymean

    @handle_arrays
    def yhat_to_ystd(self, yhat):
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute tstd
        tstd = self.t_std(a, b)

        # Compute and return ystd
        ystd = s * tstd

        return ystd
