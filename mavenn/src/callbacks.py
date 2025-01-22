import tensorflow as tf

class IVariationalCallback(tf.keras.callbacks.Callback):
    """Custom callback to track I_variational after each epoch"""

    def __init__(self, model, validation=False):
        """
        If validation=True, calculate I_variational on validation data.
        Otherwise, calculate I_variational on training data.
        """
        super().__init__()
        self.mavenn_model = model
        self.validation = validation
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate I_variational after each epoch and store in logs dict
        """
        if logs is None:
            logs = {}
        
        # Choose indices and key name based on whether validation is True
        if self.validation:
            ix = self.mavenn_model.validation_flags
            key_name = 'val_I_var'
        else:
            ix = ~self.mavenn_model.validation_flags
            key_name = 'I_var'
        
        # Set data to use 
        x = self.mavenn_model.x[ix]
        if self.mavenn_model.regression_type == 'GE':
            y = self.mavenn_model.y[ix]
        else:
            y = self.mavenn_model.y[ix, :]
        
        # Calculate I_variational and store in logs dict
        I_var, _ = self.mavenn_model.I_variational(x, y, uncertainty=False)
        logs[key_name] = I_var
