# import tensorflow as tf

# class IVarMetric(tf.keras.metrics.Metric):
#     """
#     A custom metric to track the I_var value from a noise layer.
#     """
#     def __init__(self, noise_layer, name='I_var', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.noise_layer = noise_layer
#         self.I_var = self.add_weight(name='I_var', initializer='zeros')
 
#     def update_state(self, y_true, y_pred, sample_weight=None):
        
#         # Access the Python float value that was stored during the last call()
#         self.I_var.assign(tf.reduce_mean(self.noise_layer.I_var))
        
#     def result(self):
#         return self.I_var
    