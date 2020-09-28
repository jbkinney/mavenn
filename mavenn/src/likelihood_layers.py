import numpy as np

# Tensorflow imports
import tensorflow.keras.backend as K
import tensorflow.keras
import tensorflow as tf

eps = 1E-10
pi = np.pi
e = np.exp(1)


class GaussianLikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called
    inputs. Outputs consist of negative log likelihood values.
    Computes likelihood using the Gaussian distribution.
    Layer includes 4 trainable scalar weights: w, a, b, c.
    """

    def __init__(self,
                 info_for_layers_dict,
                 polynomial_order=2,
                 eta_regularization=0.01,
                 **kwargs):

        # order of polynomial which defines log_sigma's dependence on y_hat
        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization
        self.info_for_layers_dict = info_for_layers_dict

        super(GaussianLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    def build(self, input_shape):
        self.a = self.add_weight(name='a', shape=(self.polynomial_order+1, 1),
                                 initializer="TruncatedNormal", trainable=True)

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

        # Compute logsigma from a coefficients
        self.logsigma = 0
        for poly_coeff_index in range(self.polynomial_order+1):
            if poly_coeff_index == 0:
                self.logsigma += self.a[poly_coeff_index]
            else:
                self.logsigma += self.a[poly_coeff_index] * \
                                 K.pow(yhat, poly_coeff_index)

        # Regularize parameters of the polynomials
        self.add_loss(self.eta_regularization * tf.norm(self.a) ** 2)

        # Compute negative log likelihood
        negative_log_likelihood = \
            0.5 * K.square((ytrue - yhat) / K.exp(self.logsigma)) \
            + self.logsigma \
            + 0.5*np.log(2*pi)

        # Add I_like metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = K.mean(np.log2(e)*negative_log_likelihood)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_like", aggregation="mean")

        return negative_log_likelihood


class CauchyLikelihoodLayer(tensorflow.keras.layers.Layer):

    def __init__(self,
                 info_for_layers_dict,
                 polynomial_order=2,
                 eta_regularization=0.01,
                 **kwargs):

        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization
        self.info_for_layers_dict = info_for_layers_dict

        super(CauchyLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    def build(self, input_shape):
        self.a = self.add_weight(name='a', shape=(self.polynomial_order+1, 1),
                                 initializer="TruncatedNormal", trainable=True)

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

        self.log_gamma = 0
        for poly_coeff_index in range(self.polynomial_order+1):
            if poly_coeff_index == 0:
                self.log_gamma += self.a[poly_coeff_index]
            else:
                self.log_gamma += self.a[poly_coeff_index] * \
                                  K.pow(yhat, poly_coeff_index)

        # regularize parameters of the polynomials
        self.add_loss(self.eta_regularization * tf.norm(self.a) ** 2)

        # Compute contributions to negative log likelihood
        negative_log_likelihood = \
            K.log(K.exp(2*self.log_gamma) + K.square(ytrue - yhat) + eps) \
            - self.log_gamma \
            + np.log(pi)

        # Add I_like metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = K.mean(np.log2(e)*negative_log_likelihood)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_like", aggregation="mean")

        return negative_log_likelihood


class SkewedTLikelihoodLayer(tensorflow.keras.layers.Layer):

    def __init__(self,
                 info_for_layers_dict,
                 polynomial_order=2,
                 eta_regularization=0.01,
                 **kwargs):

        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization
        self.info_for_layers_dict = info_for_layers_dict

        super(SkewedTLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "w_a": self.w_a,
                "w_b": self.w_b, "w_s": self.w_s}

    def build(self, input_shape):
        self.w_a = self.add_weight(name='w_a',
                                   shape=(self.polynomial_order+1, 1),
                                   initializer="random_normal",
                                   trainable=True)

        self.w_b = self.add_weight(name='w_b',
                                   shape=(self.polynomial_order+1, 1),
                                   initializer="random_normal",
                                   trainable=True)

        self.w_s = self.add_weight(name='w_s',
                                   shape=(self.polynomial_order+1, 1),
                                   initializer="random_normal",
                                   trainable=True)

        # Continue building keras.laerys.Layer class
        # super.build(input_shape)

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

        # To clarify equations
        Log = K.log
        LogGamma = tf.math.lgamma
        Exp = K.exp
        Sqrt = K.sqrt
        Square = K.square

        # TODO: this is throwing a warning with tensorflow 2.1.0,
        # but not with tf versions > 2.1.0
        # disabling eager_execution tf.compat.v1.disable_eager_execution()
        # may make it work in future
        # versions as a hack, but may have to use tf.gather_nd() rather than
        # sliciing manually to remove warning.

        y_hat = inputs[:, 0:1] # this is yhat

        self.log_a = 0
        self.log_b = 0
        self.log_scale = 0

        for poly_coeff_index in range(self.polynomial_order+1):
            if poly_coeff_index == 0:
                self.log_a += self.w_a[poly_coeff_index]
                self.log_b += self.w_b[poly_coeff_index]
                self.log_scale += self.w_s[poly_coeff_index]
            else:
                self.log_a += self.w_a[poly_coeff_index] * \
                              K.pow(y_hat, poly_coeff_index)
                self.log_b += self.w_b[poly_coeff_index] * \
                              K.pow(y_hat, poly_coeff_index)
                self.log_scale += self.w_s[poly_coeff_index] * \
                                  K.pow(y_hat, poly_coeff_index)

        # Compute a, b, scale in terms of trainable parameters
        self.a = Exp(self.log_a)
        self.b = Exp(self.log_b)
        self.scale = Exp(self.log_scale)

        # these are the labels
        ytrue = inputs[:, 1:]

        # Compute the mode of the unscaled, unshifted t-distribution
        self.t_mode = ((self.a - self.b) * Sqrt(self.a + self.b)) \
                      / (Sqrt(2 * self.a + 1) * Sqrt(2 * self.b + 1))

        # Compute the t value corresponding to y, assuming the mode is at y_hat
        t = self.t_mode + (ytrue - y_hat) / self.scale

        # Compute useful argument
        arg = t / Sqrt(self.a + self.b + Square(t))

        # regularize parameters of the polynomials
        self.add_loss(self.eta_regularization * tf.norm(self.w_a) ** 2)
        self.add_loss(self.eta_regularization * tf.norm(self.w_b) ** 2)
        self.add_loss(self.eta_regularization * tf.norm(self.w_s) ** 2)

        # Compute the log likelihood of y given y_hat and return
        log_likelihood = (self.a + 0.5) * Log(1 + arg) + \
                         (self.b + 0.5) * Log(1 - arg) + \
                         -(self.a + self.b - 1) * Log(2.0) + \
                         -0.5 * Log(self.a + self.b) + \
                         LogGamma(self.a + self.b) + \
                         -LogGamma(self.a) + \
                         -LogGamma(self.b) + \
                         -self.log_scale
        negative_log_likelihood = -log_likelihood

        # Add I_like metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = K.mean(np.log2(e)*negative_log_likelihood)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_like", aggregation="mean")

        return negative_log_likelihood


class MPALikelihoodLayer(tensorflow.keras.layers.Layer):

    def __init__(self,
                 info_for_layers_dict,
                 number_bins,
                 **kwargs):

        self.number_bins = number_bins
        self.info_for_layers_dict = info_for_layers_dict
        super(MPALikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        pass

    def build(self, input_shape):
        pass

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




