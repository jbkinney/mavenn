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

    def __init__(self, **kwargs):

        super(GaussianLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "w": self.w, "a": self.a,
                               "b": self.b, "c": self.c}

    def build(self, input_shape):

        # logsigma = w*sigmoid(a*y_hat + b) + c, where w, a, b, and c are trainable parameters.
        self.w = self.add_weight(name='w', shape=(1, 1), initializer="random_normal", trainable=True)
        self.a = self.add_weight(name='a', shape=(1, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(name='b', shape=(1, 1), initializer="random_normal", trainable=True)
        self.c = self.add_weight(name='c', shape=(1, 1), initializer="random_normal", trainable=True)

    def call(self, inputs):

        # compute per-datum ll here

        # this is yhat
        yhat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        self.logsigma = self. w * K.sigmoid(self. a * yhat + self.b) + self.c

        negative_log_likelihood = 0.5 * K.sum(K.square((ytrue - yhat) / K.exp(self.logsigma)) + self.logsigma, axis=1)
        return negative_log_likelihood


class CauchyLikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood using the Cauchy distribution.
    Layer includes 4 trainable scalar weights: w, a, b, c.
    """

    def __init__(self, **kwargs):

        super(CauchyLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "w": self.w, "a": self.a,
                               "b": self.b, "c": self.c}

    def build(self, input_shape):

        self.w = self.add_weight(name='w', shape=(1, 1), initializer="random_normal", trainable=True)
        self.a = self.add_weight(name='a', shape=(1, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(name='b', shape=(1, 1), initializer="random_normal", trainable=True)
        self.c = self.add_weight(name='c', shape=(1, 1), initializer="random_normal", trainable=True)

    def call(self, inputs):

        # compute negative ll here

        # this is yhat
        yhat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        self.gamma = self.w * K.sigmoid(self.a * yhat + self.b) + self.c

        # Negative log Cauchy likelihood
        negative_log_likelihood = K.sum(K.log(K.square((ytrue - yhat)) +
                                              K.square(K.exp(self.gamma))) - self.gamma, axis=1)
        return negative_log_likelihood


class SkewedTLikelihoodLayer(tensorflow.keras.layers.Layer):
    """
    Inputs consit of y and y_hat -> they are contained in a single array called inputs
    Outputs consist of negative log likelihood values.
    Computes likelihood using the skewed t-distribution proposed by Jones and Faddy (2003).
    Layer includes three trainable scalar weights: log_a, log_b, and log_scale.
    """

    def __init__(self, **kwargs):
        super(SkewedTLikelihoodLayer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "log_a": self.log_a, "log_b": self.log_b, "log_scale": self.log_scale}

    def build(self, batch_input_shape):
        # Create trainable weights
        self.log_a = self.add_weight(name="log_a", trainable=True, initializer="zeros", shape=(1, 1))
        self.log_b = self.add_weight(name="log_b", trainable=True, initializer="zeros", shape=(1, 1))
        self.log_scale = self.add_weight(name="log_scale", trainable=True, initializer="zeros", shape=(1, 1))

        # Continue building keras.laerys.Layer class
        # super.build(batch_input_shape)

    def call(self, inputs):

        # To clarify equations
        Log = K.log
        LogGamma = tf.math.lgamma
        Exp = K.exp
        Sqrt = K.sqrt
        Square = K.square

        # Compute a, b, scale in terms of trainable parameters
        self.a = Exp(self.log_a)
        self.b = Exp(self.log_b)
        self.scale = Exp(self.log_scale)

        # this is yhat
        y_hat = inputs[:, 0:1]

        # these are the labels
        ytrue = inputs[:, 1:]

        # Compute the mode of the unscaled, unshifted t-distribution
        self.t_mode = (self.a - self.b) * Sqrt(self.a + self.b) / (Sqrt(2 * self.a + 1) * Sqrt(2 * self.b + 1))

        # Compute the t value corresponding to y, assuming the mode is at y_hat
        t = self.t_mode + (ytrue - y_hat) / self.scale

        # Compute useful argument
        arg = t / Sqrt(self.a + self.b + Square(t))

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
