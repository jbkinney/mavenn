import numpy as np
import tensorflow as tf

class IVariationalCallback(tf.keras.callbacks.Callback):
    """Custom callback to track I_variational after each epoch"""

    def __init__(self, model, validation=False, N_max=10_000):
        """
        If validation=True, calculate I_variational on validation data.
        Otherwise, calculate I_variational on training data.
        """
        super().__init__()
        self.mavenn_model = model
        self.validation = validation
        self.N_max = N_max
        
        # Choose indices and key name based on whether validation is True
        if self.validation:
            ix = self.mavenn_model.validation_flags.copy()
            self.key_name = 'val_I_var'
        else:
            ix = ~self.mavenn_model.validation_flags.copy()
            self.key_name = 'I_var'
            
        # Subsample data if necessary
        if sum(ix) > self.N_max:
            # Get indices where ix is True
            true_indices = np.where(ix)[0]
            
            # Randomly choose N_max indices
            selected_indices = np.random.choice(true_indices, 
                                                size=self.N_max, 
                                                replace=False)
            
            # Create new boolean mask with only selected indices set to True
            ix = np.zeros_like(ix)
            ix[selected_indices] = True
            
        # Set data to use 
        self.x = self.mavenn_model.x[ix]
        if self.mavenn_model.regression_type == 'GE':
            self.y = self.mavenn_model.y[ix]
        else:
            self.y = self.mavenn_model.y[ix, :]
        self.num_datapoints = self.x.shape[0]
            
        # Make sure that there are at least 15 datapoints
        if self.num_datapoints < 15:
            raise AssertionError
        
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate I_variational after each epoch and store in logs dict
        """
        if logs is None:
            logs = {}
        
        # Calculate I_variational and store in logs dict
        try:
            I_var, _ = self.mavenn_model.I_variational(self.x, self.y, uncertainty=False)
            logs[self.key_name] = I_var
        except AssertionError:
            print('Debugging...')
            raise AssertionError
