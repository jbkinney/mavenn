import tensorflow.keras.backend as K
import tensorflow.keras
import tensorflow as tf


class GaussianLikelihoodLayer(tensorflow.keras.layers.Layer):

    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood using the Gaussian distribution.
    Layer includes 4 trainable scalar weights: w, a, b, c.
    """

    def __init__(self, polynomial_order=2, eta_regularization=0.01, **kwargs):

        # order of polynomial which defines log_sigma's dependence on y_hat
        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization

        super(GaussianLikelihoodLayer, self).__init__(**kwargs)

    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "w": self.w, "a": self.a,
    #                            "b": self.b, "c": self.c}

    def build(self, input_shape):

        self.a = self.add_weight(name='a', shape=(self.polynomial_order+1, 1),
                                 initializer="TruncatedNormal", trainable=True)

    def call(self, inputs):

        # this is yhat
        yhat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        # replace the tensors where nans in ytrue occur with zeros, so that likelihood for
        # that yhat, ypred pair is also zero.
        # yhat = tf.where(tf.is_nan(ytrue), tf.zeros_like(yhat), yhat)
        # ytrue = tf.where(tf.is_nan(ytrue), tf.zeros_like(ytrue), ytrue)

        self.logsigma = 0
        for poly_coeff_index in range(self.polynomial_order+1):
            self.logsigma += self.a[poly_coeff_index]*K.pow(yhat, poly_coeff_index)

        # regularize parameters of the polynomials
        self.add_loss(self.eta_regularization * tf.norm(self.a) ** 2)

        negative_log_likelihood = 0.5 * K.sum(K.square((ytrue - yhat) / K.exp(self.logsigma)) + self.logsigma, axis=1)

        #self.add_loss(negative_log_likelihood)

        return negative_log_likelihood


class CauchyLikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood using the Cauchy distribution.
    Layer includes 4 trainable scalar weights: w, a, b, c.
    """

    def __init__(self, polynomial_order=2, eta_regularization=0.01, **kwargs):

        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization

        super(CauchyLikelihoodLayer, self).__init__(**kwargs)

    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "w": self.w, "a": self.a,
    #                            "b": self.b, "c": self.c}

    def build(self, input_shape):

        self.a = self.add_weight(name='a', shape=(self.polynomial_order+1, 1),
                                 initializer="TruncatedNormal", trainable=True)

    def call(self, inputs):

        # compute negative ll here

        # this is yhat
        yhat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        self.log_gamma = 0
        for poly_coeff_index in range(self.polynomial_order+1):
            self.log_gamma += self.a[poly_coeff_index]*K.pow(yhat, poly_coeff_index)

        # regularize parameters of the polynomials
        self.add_loss(self.eta_regularization * tf.norm(self.a) ** 2)

        # Negative log Cauchy likelihood
        negative_log_likelihood = K.sum(K.log(K.square((ytrue - yhat)) +
                                              K.square(K.exp(self.log_gamma))) - self.log_gamma, axis=1)
        return negative_log_likelihood


class SkewedTLikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood using the skewed t-distribution proposed by Jones and Faddy (2003).
    Layer includes three trainable scalar weights: log_a, log_b, and log_scale.
    """

    def __init__(self, polynomial_order=2, eta_regularization=0.01, **kwargs):

        # order of polynomial which defines the spatial parameters' dependence on y_hat
        self.polynomial_order = polynomial_order
        self.eta_regularization = eta_regularization

        super(SkewedTLikelihoodLayer, self).__init__(**kwargs)

    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "log_a": self.log_a, "log_b": self.log_b, "log_scale": self.log_scale}

    def build(self, input_shape):


        self.w_a = self.add_weight(name='w_a', shape=(self.polynomial_order+1, 1),
                                 initializer="random_normal", trainable=True)

        self.w_b = self.add_weight(name='w_b', shape=(self.polynomial_order+1, 1),
                                 initializer="random_normal", trainable=True)

        self.w_s = self.add_weight(name='w_s', shape=(self.polynomial_order+1, 1),
                                 initializer="random_normal", trainable=True)

        # Continue building keras.laerys.Layer class
        # super.build(batch_input_shape)

    def call(self, inputs):

        # To clarify equations
        Log = K.log
        LogGamma = tf.math.lgamma
        Exp = K.exp
        Sqrt = K.sqrt
        Square = K.square

        # TODO: this is throwing a warning with tensorflow 2.1.0, but not with tf versions > 2.1.0
        # disabling eager_execution tf.compat.v1.disable_eager_execution() may make it work in future
        # versions as a hack, but may have to use tf.gather_nd() rather than sliciing manually to remove warning.

        y_hat = inputs[:, 0:1] # this is yhat

        self.log_a = 0
        self.log_b = 0
        self.log_scale = 0

        for poly_coeff_index in range(self.polynomial_order+1):

            self.log_a += self.w_a[poly_coeff_index]*K.pow(y_hat, poly_coeff_index)
            self.log_b += self.w_b[poly_coeff_index] * K.pow(y_hat, poly_coeff_index)
            self.log_scale += self.w_s[poly_coeff_index] * K.pow(y_hat, poly_coeff_index)

        # Compute a, b, scale in terms of trainable parameters
        self.a = Exp(self.log_a)
        self.b = Exp(self.log_b)
        self.scale = Exp(self.log_scale)

        # these are the labels
        ytrue = inputs[:, 1:]

        # Compute the mode of the unscaled, unshifted t-distribution
        self.t_mode = (self.a - self.b) * Sqrt(self.a + self.b) / (Sqrt(2 * self.a + 1) * Sqrt(2 * self.b + 1))

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

        return -log_likelihood


class MPALikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood for NAR.
    """

    def __init__(self, number_bins, **kwargs):

        self.number_bins = number_bins
        super(MPALikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):

        pass

    def build(self, input_shape):

        pass

    def call(self, inputs):

        # this is yhat
        p_y_given_phi = inputs[:, 0:self.number_bins]

        # these are the labels
        c_my = inputs[:, self.number_bins:]

        # one sum over y, other sum over m
        return -K.sum(K.sum(c_my * K.log(p_y_given_phi), axis=1), axis=0)
        #return -K.sum(c_my * K.log(p_y_given_phi))


