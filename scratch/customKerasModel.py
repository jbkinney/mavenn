import tensorflow as tf
from tensorflow import keras

class CustomKerasModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize a metric for information
        self.I_var = keras.metrics.Mean(name='I_var')
        
    def on_epoch_end(self, epoch, logs=None):
        # Update I_var metric at the end of each epoch
        if logs is None:
            logs = {}
        logs['I_var'] = tf.keras.backend.get_value(self.layers[-1].I_var)
        self.I_var.update_state(logs['I_var'])
        
        
    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Compute predictions
        y_pred = self(x, training=False)
        
        # Update loss
        self.compiled_loss(y, y_pred, sample_weight=sample_weight)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(self.compiled_loss(y, y_pred, sample_weight=sample_weight))
            elif isinstance(metric, keras.metrics.Mean):
                metric.update_state(y_pred)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
        
        # Get results dictionary
        results = {m.name: m.result() for m in self.metrics}
        
        # Add I_var to validation metrics
        results['I_var'] = tf.keras.backend.get_value(self.layers[-1].I_var)
        return results

    def train_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
            
        # Run forward pass inside gradient tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            #loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
            loss = self.compute_loss(x, y, y_pred, sample_weight=sample_weight, training=True)
                       
        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics in a loop
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            elif isinstance(metric, keras.metrics.Mean):
                # Mean metrics only take a single value
                metric.update_state(y_pred)
            else:
                # Other metrics may take multiple arguments
                metric.update_state(y, y_pred, sample_weight=sample_weight)
        
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        
        return results