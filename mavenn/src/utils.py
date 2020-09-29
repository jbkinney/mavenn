# Standard imports
import numpy as np
import pandas as pd
import mavenn
import pdb
import pickle

# Special function imports
from numpy import log as Log
from scipy.special import betaincinv as BetaIncInv
from scipy.special import gammaln as LogGamma
from scipy.stats import cauchy
from scipy.special import erfinv

# Imports from MAVE-NN
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_seqs, validate_1d_array, validate_alphabet


class SkewedTNoiseModel:

    """
    Class used to compute quantiles for the Skewed T noise model for
    GE regression. Usage example:

    quantiles = ComputeSkewedTQuantiles(GER,yhat_GE)

    The attributes gives +/- 1 sigma quantiles

    quantiles.plus_sigma_quantile
    quantiles.minus_sigma_quantile

    parameters
    ----------

    model: (mavenn.Model object)

         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Skewed T noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the quantiles will be computed.
        This should be the output of the GE model.


    user_quantile: (float between [0,1])
        If not None, the attribute user_quantile_values will contain quantile values
        for the user_quantile value specified
    """

    def __init__(self,
                 model,
                 yhat_GE,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat_GE = yhat_GE
        self.q = q

        polynomial_weights_a = self.model.get_nn().layers[6].get_weights()[0].copy()
        polynomial_weights_b = self.model.get_nn().layers[6].get_weights()[1].copy()
        polynomial_weights_s = self.model.get_nn().layers[6].get_weights()[2].copy()

        log_a = 0
        log_b = 0
        log_scale = 0

        for polynomial_index in range(len(polynomial_weights_a)):
            log_a += polynomial_weights_a[polynomial_index][0] * np.power(yhat_GE, polynomial_index)
            log_b += polynomial_weights_b[polynomial_index][0] * np.power(yhat_GE, polynomial_index)
            log_scale += polynomial_weights_s[polynomial_index][0] * np.power(yhat_GE, polynomial_index)

        self.plus_sigma_quantile = self.y_quantile(0.16, self.yhat_GE, np.exp(log_scale), np.exp(log_a), np.exp(log_b))
        self.minus_sigma_quantile = self.y_quantile(0.84, self.yhat_GE, np.exp(log_scale), np.exp(log_a), np.exp(log_b))

        self.log_a = log_a
        self.log_b = log_b
        self.log_scale = log_scale

        self.a = np.exp(self.log_a)
        self.b = np.exp(self.log_b)
        self.s = np.exp(self.log_scale)

        if q is not None:

            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(self.y_quantile(current_q,
                                                                 self.yhat_GE,
                                                                 np.exp(self.log_scale),
                                                                 np.exp(self.log_a),
                                                                 np.exp(self.log_b)).ravel())


    # First compute log PDF to avoid overflow problems
    def log_f(self, t, a, b):
        arg = t / np.sqrt(a + b + t ** 2)
        return (1 - a - b) * Log(2) + \
               -0.5 * Log(a + b) + \
               LogGamma(a + b) + \
               -LogGamma(a) + \
               -LogGamma(b) + \
               (a + 0.5) * Log(1 + arg) + \
               (b + 0.5) * Log(1 - arg)


    # Exponentiate to get true distribution
    def f(self, t, a, b):
        return np.exp(self.log_f(t, a, b))


    # Compute the mode as a function of a and b
    def t_mode(self, a, b):
        return (a - b) * np.sqrt(a + b) / (np.sqrt(2 * a + 1) * np.sqrt(2 * b + 1))


    # Compute mean
    def t_mean(self, a, b):
        if a <= 0.5 or b <= 0.5:
            return np.nan
        else:
            return 0.5 * (a - b) * np.sqrt(a + b) * np.exp(
                LogGamma(a - 0.5) + LogGamma(b - 0.5) - LogGamma(a) - LogGamma(b))


    # Compute variance
    def t_std(self, a, b):
        if a <= 1 or b <= 1:
            return np.nan
        else:
            t_expected = self.t_mean(a, b)
            tsq_expected = 0.25 * (a + b) * ((a - b) ** 2 + (a - 1) + (b - 1)) / ((a - 1) * (b - 1))
            return np.sqrt(tsq_expected - t_expected ** 2)


    def p_of_y_given_yhat(self,
                          y,
                          y_mode):
                          # y_scale,
                          # a,
                          # b):

        # t = self.t_mode(a, b) + (y - y_mode) / y_scale
        # return self.f(t, a, b) / y_scale

        y_scale = np.exp(self.log_scale)
        a = np.exp(self.log_a)
        b = np.exp(self.log_b)

        t = self.t_mode(a, b) + (y - y_mode) / y_scale
        return self.f(t, a, b) / y_scale


    def p_of_y_given_phi(self,
                         y,
                         phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)


    def p_mean_std(self, y_mode, y_scale, a, b):
        y_mean = self.t_mean(a, b) * y_scale + y_mode
        y_std = self.t_std(a, b) * y_scale
        return y_mean, y_std


    def t_quantile(self, q, a, b):
        x_q = BetaIncInv(a, b, q)
        t_q = (2 * x_q - 1) * np.sqrt(a + b) / np.sqrt(1 - (2 * x_q - 1) ** 2)
        return t_q


    def y_quantile(self, q, y_hat, s, a, b):
        t_q = self.t_quantile(q, a, b)
        y_q = (t_q - self.t_mode(a, b)) * s + y_hat
        return y_q

    def draw_y_values(self):
        """Draws y values, one for each value in yhat_GE"""
        N = len(self.yhat_GE)
        q = np.random.rand(N)
        t_q = self.t_quantile(q, self.a, self.b)
        y = (t_q - self.t_mode(self.a, self.b)) * self.s + self.yhat_GE
        return y


@handle_errors
class GaussianNoiseModel:
    """
    Class used to obtain +/- sigma from the Gaussian noise model
    in the GE model. The sigma, which is a function of y_hat, can
    be used to plot confidence intervals around y_hat.


    model: (mavenn.Model object)
         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Gaussian noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the confidence intervals will be computed.
        This should be the output of the GE model.

    """


    def __init__(self,
                 model,
                 yhat_GE,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat_GE = yhat_GE

        self.polynomial_weights = self.model.get_nn().layers[6].get_weights()[0].copy()
        logsigma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            logsigma += self.polynomial_weights[polynomial_index][0] * np.power(yhat_GE, polynomial_index)

        # this is sigma(y)
        self.sigma = np.exp(logsigma)

        if q is not None:
            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(yhat_GE+self.sigma*np.sqrt(2)*erfinv(2*current_q-1))

            #self.user_quantile_values = np.array(self.user_quantile_values).reshape(len(yhat_GE), len(q))


    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # recompute logsimga here instead of using self.gamma since y,yhat input
        # to this method could be different than init
        logsigma = 0

        for polynomial_index in range(len(self.polynomial_weights)):
            logsigma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        sigma = np.exp(logsigma)

        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((y - yhat) ** 2) / (2 * sigma ** 2))


    def p_of_y_given_phi(self,
                       y,
                       phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        print(y_hat_of_phi.shape)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)

    def draw_y_values(self):
        """Draws y values, one for each value in yhat_GE"""
        N = len(self.yhat_GE)
        q = np.random.rand(N)
        y = self.yhat_GE + self.sigma * np.sqrt(2) * erfinv(2 * q - 1)
        return y


@handle_errors
class CauchyNoiseModel:

    """
    Class used to obtain +/- sigma from the Cauchy noise model
    in the GE model. The sigma, which is a function of y_hat, can
    be used to plot confidence intervals around y_hat.


    model: (mavenn.Model object)
         This is the mavenn model object instantiated as a GE model. The weights
         of the polynomials for the computation of the spatial parameters of the
         Cauchy noise models are extracted from this object.

    yhat_GE: (array-like)
        This is the array of points on which the confidence intervals will be computed.
        This should be the output of the GE model.

    """

    def __init__(self,
                 model,
                 yhat_GE,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat = yhat_GE
        self.q = q

        self.polynomial_weights = self.model.get_nn().layers[6].get_weights()[0].copy()

        self.log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            self.log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat_GE, polynomial_index)

        self.plus_sigma_quantile = self.y_quantile(0.16, self.yhat)
        self.minus_sigma_quantile = self.y_quantile(0.84, self.yhat)
        self.gamma = np.exp(self.log_gamma)

        if q is not None:
            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(self.y_quantile(current_q, self.yhat).ravel())


    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # recompute gamma here instead of using self.gamma since y,yhat input
        # to this method could be different than init
        log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        return cauchy(loc=yhat, scale=np.exp(log_gamma)).pdf(y)


    def p_of_y_given_phi(self,
                         y,
                         phi):
        """
        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        phi: (array-like of floats)
            The latent phenotype values of which the probability
            probability density will be conditioned.

        """

        y_hat_of_phi = self.model.phi_to_yhat(phi)
        return self.p_of_y_given_yhat(y, y_hat_of_phi)


    def y_quantile(self,
                   user_quantile,
                   yhat):

        """
        user_quantile: (float between [0,1])
            The value representing the quantile which will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.
        """

        # compute gamma for the yhat entered
        log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        return cauchy(loc=yhat, scale=np.exp(log_gamma)).ppf(user_quantile)

    def draw_y_values(self):
        """Draws y values, one for each value in yhat_GE"""
        N = len(self.yhat)
        q = np.random.rand(N)
        y = self.yhat + self.gamma * np.tan(np.pi * (q - 0.5))
        return y


@handle_errors
def load(filename, verbose=True):

        """
        Method that will load a mave-nn model

        parameters
        ----------
        filename: (str)
            Filename of saved model.

        verbose: (bool)
            Whether to provide user feedback.

        returns
        -------
        loaded_model (mavenn-Model object)
            The model object that can be used to make predictions etc.

        """

        # Load model
        filename_pickle = filename + '.pickle'
        with open(filename_pickle, 'rb') as f:
            config_dict = pickle.load(f)

        # Create model object
        loaded_model = mavenn.Model(**config_dict['model_kwargs'])

        # Add in diffeomorphic mode fixing and standardization params
        loaded_model.unfixed_phi_mean = config_dict.get('unfixed_phi_mean', 0)
        loaded_model.unfixed_phi_std = config_dict.get('unfixed_phi_std', 1)
        loaded_model.y_mean = config_dict.get('y_mean', 0)
        loaded_model.y_std = config_dict.get('y_std', 1)
        loaded_model.x_stats = config_dict.get('x_stats', {})
        loaded_model.history = config_dict.get('history', None)
        loaded_model.info_for_layers_dict = \
            config_dict.get('info_for_layers_dict', {})

        # Load and set weights
        filename_h5 = filename + '.h5'
        loaded_model.get_nn().load_weights(filename_h5)

        # Provide feedback
        if verbose:
            print(f'Model loaded from these files:\n'
                  f'\t{filename_pickle}\n'
                  f'\t{filename_h5}')

        # Return model
        return loaded_model


@handle_errors
def vec_data_to_mat_data(y_n,
                         ct_n=None,
                         x_n=None):
    """

    Function to transform from vector to matrix format for MPA
    regression and MPA model evaluation.

    parameters
    ----------

    y_n: (np.ndarray)
        Array of N bin numbers y. Must be set by user.

    ct_n: (np.ndarray)
        Array N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (np.ndarray)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    returs
    ------

    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.
    """

    # Note: this use of validate_1d_array is needed to avoid a subtle
    # bug that occurs when inputs are pandas series with non-continguous
    # indices
    y_n = validate_1d_array(y_n).astype(int)
    N = len(y_n)
    if x_n is not None:
        x_n = validate_1d_array(x_n)
    else:
        x_n = np.arange(N)

    if ct_n is not None:
        ct_n = validate_1d_array(ct_n).astype(int)
    else:
        ct_n = np.ones(N).astype(int)

    # Cast y as array of ints
    y_n = np.array(y_n).astype(int)
    N = len(x_n)

    # This case is only for loading data. Should be tested/made more robust
    if N == 1:
        # should do check like check(mavenn.load_model==True,...

        return y_n.reshape(-1, y_n.shape[0]), x_n

    # Create dataframe
    data_df = pd.DataFrame()
    data_df['x'] = x_n
    data_df['y'] = y_n
    data_df['ct'] = ct_n

    # Sum over repeats
    data_df = data_df.groupby(['x', 'y']).sum().reset_index()

    # Pivot dataframe
    data_df = data_df.pivot(index='x', columns='y', values='ct')
    data_df = data_df.fillna(0).astype(int)

    # Clean dataframe
    data_df.reset_index(inplace=True)
    data_df.columns.name = None

    # Get ct_my values
    cols = [c for c in data_df.columns if not c in ['x']]

    ct_my = data_df[cols].values.astype(int)

    # Get x_m values
    x_m = data_df['x'].values

    return ct_my, x_m
