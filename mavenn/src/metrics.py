import tensorflow as tf

class IVarMetric(tf.keras.metrics.Metric):
    model = None
    
    def __init__(self, model, name='I_var', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.I_var = self.add_weight(name='I_var', initializer='zeros')

        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get I_var from the measurement process layer
        # Assuming your layer is accessible through the model
        try:
            self.I_var.assign(self.model.I_var)
        except AttributeError:
            self.I_var.assign(0.0)
        
    def result(self):
        return self.I_var
    
    def reset_state(self):
        pass