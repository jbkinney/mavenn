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


@handle_errors
def get_gpmap_params_in_cannonical_gauge(model):

    # Get basic dimension info on model
    L = model.L
    C = model.C
    alphabet = model.alphabet

    # Get constant and additive parameters separately
    df_0 = model.get_gpmap_parameters(which='constant', fix_gauge=False)
    df_lc = model.get_gpmap_parameters(which='additive', fix_gauge=False)
    if model.gpmap_type != "additive":
        df_lclc = model.get_gpmap_parameters(which='pairwise', fix_gauge=False)

    # Make copies for fixed values. Only need name and value columns
    df_0_new = df_0.copy()[['name', 'value']]
    df_lc_new = df_lc.copy()[['name', 'value']]
    if model.gpmap_type != "additive":
        df_lclc_new = df_lclc.copy()[['name', 'value']]

    # Compute new constant term
    df_0_new.loc[:, 'value'] += (1/C) * df_lc.loc[:, 'value'].sum()
    if model.gpmap_type != "additive":
        df_0_new.loc[:, 'value'] += (1/(C**2)) * df_lclc.loc[:, 'value'].sum()

    # Compute new additive terms
    for l in range(L):
        ix_l = (df_lc['l'] == l)
        df_lc_new.loc[ix_l, 'value'] += -(1/C) * df_lc.loc[ix_l, 'value'].sum()
        if model.gpmap_type != "additive":
            for c in alphabet:
                ix_lc = (df_lc['l'] == l) & (df_lc['c'] == c)
                ix_lcxx = (df_lclc['l1'] == l) & (df_lclc['c1'] == c)
                ix_xxlc = (df_lclc['l2'] == l) & (df_lclc['c2'] == c)
                ix_lxxx = (df_lclc['l1'] == l)
                ix_xxlx = (df_lclc['l2'] == l)

                df_lc_new.loc[ix_lc, 'value'] += \
                    (1/C) * df_lclc.loc[ix_lcxx, 'value'].sum() + \
                    (1/C) * df_lclc.loc[ix_xxlc, 'value'].sum()
                df_lc_new.loc[ix_lc, 'value'] += \
                    -(1/C**2) * df_lclc.loc[ix_lxxx, 'value'].sum() + \
                    -(1/C**2) * df_lclc.loc[ix_xxlx, 'value'].sum()

    if model.gpmap_type != "additive":
        for l1 in range(L):
            for c1 in alphabet:
                if model.gpmap_type == "neighbor":
                    l2_range = [l1+1]
                elif model.gpmap_type == "pairwise":
                    l2_range = range(l1+1, L)
                else:
                    assert False, 'This should not happen'

                for l2 in l2_range:
                    for c2 in alphabet:
                        ix_lclc = (df_lclc['l1'] == l1) & \
                                  (df_lclc['c1'] == c1) & \
                                  (df_lclc['l2'] == l2) & \
                                  (df_lclc['c2'] == c2)
                        ix_lxlc = (df_lclc['l1'] == l1) & \
                                  (df_lclc['l2'] == l2) & \
                                  (df_lclc['c2'] == c2)
                        ix_lclx = (df_lclc['l1'] == l1) & \
                                  (df_lclc['c1'] == c1) & \
                                  (df_lclc['l2'] == l2)
                        ix_lxlx = (df_lclc['l1'] == l1) & \
                                  (df_lclc['l2'] == l2)

                        df_lclc_new.loc[ix_lclc, 'value'] += \
                            -(1/C) * df_lclc.loc[ix_lxlc, 'value'].sum() + \
                            -(1/C) * df_lclc.loc[ix_lclx, 'value'].sum()
                        df_lclc_new.loc[ix_lclc, 'value'] += \
                            (1/C**2) * df_lclc.loc[ix_lxlx, 'value'].sum()

    # Concatenate into output and return
    if model.gpmap_type != "additive":
        df_new = pd.concat([df_0_new, df_lc_new, df_lclc_new])
    else:
        df_new = pd.concat([df_0_new, df_lc_new])

    # Reset index
    df_new.reset_index(inplace=True, drop=True)

    return df_new


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

        if q is not None:

            self.user_quantile_values = []
            for current_q in q:
                self.user_quantile_values.append(self.y_quantile(current_q,
                                                                 self.yhat_GE,
                                                                 np.exp(self.log_scale),
                                                                 np.exp(self.log_a),
                                                                 np.exp(self.log_b)).ravel())

            # self.user_quantile_values = self.y_quantile(self.q,
            #                                             self.yhat_GE,
            #                                             np.exp(log_scale),
            #                                             np.exp(log_a),
            #                                             np.exp(log_b))


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


    def estimate_predictive_information(self,
                                        y,
                                        yhat,
                                        y_scale,
                                        a,
                                        b):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))
        N = len(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat, y_scale, a, b)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat, y_scale, a, b).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))

        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)

        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)
        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


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

    # TODO:
    '''
    arguments should be x, y. 
    1) map x to phi
    2) p_y_given_phi
    2) then compute mutual information via 
    '''
    def estimate_predictive_information(self,
                                        y,
                                        yhat):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat)))
        N = len(np.log2(self.p_of_y_given_yhat(y, yhat)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))

        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)
        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)

        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


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
                 yhat,
                 q=[0.16, 0.84]):

        self.model = model
        self.yhat = yhat
        self.q = q

        self.polynomial_weights = self.model.get_nn().layers[6].get_weights()[0].copy()

        self.log_gamma = 0
        for polynomial_index in range(len(self.polynomial_weights)):
            self.log_gamma += self.polynomial_weights[polynomial_index][0] * np.power(yhat, polynomial_index)

        self.plus_sigma_quantile = self.y_quantile(0.16, self.yhat)
        self.minus_sigma_quantile = self.y_quantile(0.84, self.yhat)

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


    def estimate_predictive_information(self,
                                        y,
                                        yhat):
        """
        Method that estimates predictive information, i.e.
        I[y;y_hat] = I[y;phi].

        parameters
        ----------
        y: (array-like of floats)
            y values for which the probability will be computed

        y_hat: (array-like of float)
            The y-hat values on which the probability distribution
            will be conditioned on.

        returns
        -------
        I: (float)
            Mutual information between I[y;y_hat], or equivalently
            I[y;phi]

        dI: (float)
            Error in the estimated, mutual information I
        """

        # compute log_2 of p_y_given_yhat for all values and take mean:
        mean_log_2_p_y_given_yhat = np.mean(np.log2(self.p_of_y_given_yhat(y, yhat)))

        N = len(np.log2(self.p_of_y_given_yhat(y, yhat)))
        std_log_2_p_y_given_yhat = np.std(np.log2(self.p_of_y_given_yhat(y, yhat)))/np.sqrt(N)

        p_y = []
        for _ in range(len(y)):
            '''
            form p_y by averaging over y_hat for every value of y_test
            i.e. 
            # p(y_1|y_hat_1), p(y_1|y_hat_1), ... ,p(y_1|y_hat_N), the mean of this is p(y_1)
            # p(y_2|y_hat_1), p(y_2|y_hat_1), ... ,p(y_2|y_hat_N), the mean of this is p(y_2), and so on.
            '''
            p_y.append(np.mean(self.p_of_y_given_yhat(y[_], yhat).ravel()))

        p_y = np.array(p_y)
        mean_log_2_p_y = np.mean(np.log2(p_y))
        std_log_2_p_y = np.std(np.log2(p_y))/np.sqrt(N)

        dI = np.sqrt(std_log_2_p_y_given_yhat ** 2 + std_log_2_p_y ** 2)

        I = mean_log_2_p_y_given_yhat-mean_log_2_p_y

        return I, dI


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

        # Add in diffeomorphic mode fixing params
        loaded_model.unfixed_phi_mean = config_dict['unfixed_phi_mean']
        loaded_model.unfixed_phi_std = config_dict['unfixed_phi_std']
        loaded_model.y_mean = config_dict['y_mean']
        loaded_model.y_std = config_dict['y_std']

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

    y_n: (array-like of ints)
        List of N bin numbers y. Must be set by user.

    ct_n: (array-like of ints)
        List N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (array-like)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    returs
    ------

    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.
    """

    # Cast y as array of ints
    y_n = np.array(y_n).astype(int)
    N = len(x_n)

    # Cast x as array and get length
    if x_n is None:
        x_n = np.arange(N)
    else:
       x_n = np.array(x_n)
       #assert len(x_n) == N, f'len(y_n)={len(y_n)} and len(x_n)={N} do not match.'

    # Get ct
    if ct_n is None:
        ct_n = np.ones(N).astype(int)
    #else:
        #assert len(ct_n) == N, f'len(ct_n)={len(ct_n)} and len(x_n)={N} do not match.'

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
    data_df = data_df.groupby(['x','y']).sum().reset_index()

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
