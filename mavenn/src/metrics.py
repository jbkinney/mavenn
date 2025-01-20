import tensorflow as tf

class IVarMetric(tf.keras.metrics.Metric):
    
    def __init__(self, model, name='I_var', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.I_var = self.model._layers[-1].I_var
 
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get I_var from the measurement process layer
        # Assuming your layer is accessible through the model
        pass
        
    def result(self):
        return self.I_var
    
    def reset_state(self):
        pass