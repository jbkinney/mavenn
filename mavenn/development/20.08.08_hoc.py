# Note: in all these computations, batch size is in the first index.
import tf.math.exp as Exp
import tf.math.log as Log
import tf.math.square as Square
import tf.reduce_sum as Sum

class HOCLayer(kears.layers.Layer):
    def __init__(self, N, **kwargs):
        """
        N: number of examples in training data
        """
        super().__init__(**kwargs)
        self.N = N

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "N":self.N}

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape([1])

    def build(self, batch_input_shape):

        # Add trainable random effects std: log sigma_hoc
        self.log_sigma_hoc = self.add_weight(
            name='log_sigma_hoc',
            shape=[1],
            initializer="zeros",
            trainable=True)

        # Add trainable random effects: eta
        self.eta = self.add_weight(
            name = 'eta',
            shape = [self.N],
            initializer="zeros",
            trainable=True)

        # Continue building keras.laerys.Layer class
        super.build(batch_input_shape)

    def call(self, phi_hat, indices=None):
        """ 
        Addes a (trainable) random effect eta to each input phi
        Note that each example has a different random effect.

        Then adds a term to the loss function accounting for the 
        probability of these random effects.
        """

        # Compute std dev
        sigma_hoc = Exp(self.log_sigma_hoc)

        # If indices are defined, extract corresponding random effect weights.
        if indices is not None:
            eta = tf.gather(params=self.eta, indices=indices)

        # If indices are not defined, use random numbers
        else:
            eta = tf.random(phi.shape(), stddev=sigma_hoc)

        # Compute and record contribution to loss function
        loss = 0.5*Sum(Square(eta))/Square(sigma_hoc) + \
               self.N*Log(sigma_hoc)
        self.add_loss(loss)

        # Compute updated phi values
        phi = phi_hat + eta 

        # Return updated phi
        return phi

