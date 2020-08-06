# Note: in all these computations, batch size is in the first index.

class GPLayer(keras.layers.Layer):
    def __init__(self, 
                 L, # Sequence length 
                 C, # Alphabet size
                 **kwargs):
        self.C = C 
        self.L = L 
        super().__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "L":self.L, "C":self.C}

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape([1])

# Create class for an additive GPMap specifically
class AdditiveGPLayer(GPLayer):
        
    # Redefine build
    def build(self, batch_input_shape):
        # Make sure batch_input_shape is the right length
        assert len(batch_input_shape)==2, \
            f"batch_input_shape={batch_input_shape}; should have len 2."

        # Set batch_size
        self.batch_size = batch_input_shape[0]

        # Set x_length and make sure it matches L and C
        self.X_length == batch_input_shape[1]
        assert self.X_length == self.L*self.C, \
            f'Size mismatch. x_length={x_length}. C={self.C}. L={self.L}'

        # Build constant weight
        self.theta_0 = self.add_weight(
            name='theta_0',
            shape=[1],
            initializer="glorot_normal",
            trainable=True)

        # Build additive weights
        self.theta_lc = self.add_weight(
            name='theta_l:c',
            shape=[self.L, self.C],
            initializer="glorot_normal",
            trainable=True)

        # Continue building keras.laerys.Layer class
        super.build(batch_input_shape)

    def call(self, X):
        # Create sequence matrix
        x_lc = tf.reshape(X, [self.batch_size, self.L, self.C])

        # Compute phi
        phi = self.theta_0 + \
            tf.math.reduce_sum(self.theta_lc*x_lc, axis=[1,2])
        
        # Return phi
        return phi

    def fix_gauge_modes(self, X_wt=None):
        """
        If X_wt = None, the hierarchical gauge is used
        If X_wt is set to a one-hot encoded sequence, the wildtype gauge is used
        """

        # If using the hierarchicahl gauge
        if X_wt is None:

            # Fix constant term
            self.theta_0 += (1/self.C)*tf.reduce_sum(self.theta_lc, axis=[0,1])

            # Fix additive terms
            self.theta_lc -= (1/self.C)*tf.reduce_sum(self.theta_lc, axis=1, keepdims=True)

        # If using the wildtype gauge
        else:

            # Create sequence matrix, with no batch dimension
            x_lc = tf.reshape(X_wt, [self.L, self.C])

            # Fix constant term
            self.theta_0 += tf.reduce_sum(x_lc*self.theta_lc, axis=[0,1])

            # Fix additive terms
            self.theta_lc -= tf.reduce_sum(x_lc*self.theta_lc, axis=1, keepdims=True)

def GELayer(keras.layers.Layer):
    def __init__(self, 
                 monotonic=True, 
                 hidden_nodes=10, 
                 **kwargs):
        self.monotonic = monotonic
        self.hidden_nodes = hidden_nodes
        super().__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "monotonic":self.monotonic,
                "hidden_nodes":self.hidden_nodes}

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape([1])

            # Redefine build
    def build(self, batch_input_shape):
        # Make sure batch_input_shape is the right length
        assert len(batch_input_shape)==1, \
            f"batch_input_shape={batch_input_shape}; should have len 1."

        # Set batch_size
        self.batch_size = batch_input_shape[0]

        if self.monotonic
            constraint=tf.keras.constraints.non_neg()
        else:
            constraint=None

        # Build constant weight
        self.phi_weights = self.add_weight(
            name='phi_weights',
            shape=[self.hidden_nodes],
            initializer="glorot_normal",
            trainable=True,
            constraint=constraint)

        # Build constant weight
        self.phi_biases = self.add_weight(
            name='phi_biases',
            shape=[self.hidden_nodes],
            initializer="glorot_normal",
            trainable=True)

        self.sigmoid_weights = self.add_weight(
            name='sigmoid_weights',
            shape=[self.hidden_nodes],
            initializer="glorot_normal",
            trainable=True,
            constraint=constraint)

        self.sigmoid_bias = self.add_weight(
            name='bias',
            shape=[1],
            initializer="glorot_normal",
            trainable=True)

        # Continue building keras.laerys.Layer class
        super.build(batch_input_shape)

    def call(self, X):
        # Compute output of hidden layer
        hidden_layer_out = tf.math.sigmoid(self.phi_weights*X + self.phi_biases)
        
        # Compute y_hat
        y_hat = self.sigmoid_bias + tf.math.reduce_sum(self.sigmoid_weights*hidden_layer_out) 
        
        # Return phi
        return phi

