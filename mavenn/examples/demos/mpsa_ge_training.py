#! /usr/bin/env python
"""
run_demo: mpsa_ge_training

Trains a neighbor G-P map, using GE regression on data from
Wong et al., 2018. Takes ~30 seconds to run.
"""

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Insert path to local mavenn beginning of path
import os
import sys
abs_path_to_mavenn = os.path.abspath('../../../')
sys.path.insert(0, abs_path_to_mavenn)

# Load mavenn
import mavenn
print(f'Using mavenn at: {mavenn.__path__[0]}')

# Load dataset as a dataframe
data_df = mavenn.load_example_dataset('mpsa')

# Split into training and test data
ix = (data_df['set']!='test')
train_df = data_df[ix]
test_df = data_df[~ix]

# Extract x and y as np.arrays
x_train = train_df['x'].values
y_train = train_df['y'].values
x_test = test_df['x'].values
y_test = test_df['y'].values


# Define a model with:
# - a pairwise G-P map
# - a GE measurement process
# - a heteroskedastic SkewedT noise model
model = mavenn.Model(regression_type='GE',
                     L=len(x_train[0]),
                     gpmap_type='pairwise',
                     alphabet='rna',
                     ge_noise_model_type='SkewedT',
                     ge_nonlinearity_monotonic=True,
                     ge_heteroskedasticity_order=2)

# Set training data
model.set_data(x=x_train,
               y=y_train)

# Fit model to training data
start_time = time.time()
model.fit(epochs=30,
          learning_rate=0.005,
          early_stopping=False)
training_time = time.time()-start_time

# Predict latent phentoype values (phi) on test data
phi_test = model.x_to_phi(x_test)

# Predict measurement values (yhat) on test data
yhat_test = model.x_to_yhat(x_test)

# Compute R^2 between yhat and y_test
Rsq = np.corrcoef(yhat_test.ravel(), y_test)[0, 1]**2

# Set phi lims and create grid in phi space
phi_lim = [min(phi_test)-.5, max(phi_test)+.5]
phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)

# Compute yhat each phi gridpoint
yhat_grid = model.phi_to_yhat(phi_grid)

# Compute 68% CI for each yhat
yqs_grid = model.yhat_to_yq(yhat_grid, q=[0.16, 0.84])

# Extract training loss and validation loss
loss_training = model.history['loss']
loss_validation = model.history['val_loss']

# Create figure and axes
fig, axs = plt.subplots(1, 3, figsize=[12, 4])

# Left panel: illustrate measurement process (y vs. phi)
ax = axs[0]
ax.scatter(phi_test, y_test, color='C0', s=5, alpha=.2, label='test data')
ax.plot(phi_grid, yhat_grid, linewidth=2, color='C1',
        label='$\hat{y} = g(\phi)$')
ax.plot(phi_grid, yqs_grid[:, 0], linestyle='--', color='C1', label='68% CI')
ax.plot(phi_grid, yqs_grid[:, 1], linestyle='--', color='C1')
ax.set_xlim(phi_lim)
ax.set_xlabel('latent phenotype ($\phi$)')
ax.set_ylabel('measurement ($y$)')
ax.set_title('measurement process')
ax.legend()

# Center panel: illustrate model performance (y vs. yhat)
ax = axs[1]
ys = np.vstack([y_test])
ax.scatter(yhat_test, y_test, color='C0', s=5, alpha=.2, label='test data')
lims = ax.get_xlim()
ax.plot(lims, lims, linestyle=':', color='k', label='$y=\hat{y}$')
ax.set_xlabel('model prediction ($\hat{y}$)')
ax.set_ylabel('measurement ($y$)')
ax.set_title(f'performance ($R^2$={Rsq:.3})')
ax.legend()

# Right panel: Plot model training history
ax = axs[2]
# Compute likelihood information
I_like, dI_like =  model.I_likelihood(x=x_test, y=y_test)
print(f'I_like_test: {I_like:.3f} +- {dI_like:.3f} bits')

# Compute predictive information
I_pred, dI_pred = model.I_predictive(x=x_test, y=y_test)
print(f'I_pred_test: {I_pred:.3f} +- {dI_pred:.3f} bits')

# Get training history
I_like_hist = model.history['I_like']
val_I_like_hist = model.history['val_I_like']

# Plot training history as well as information values
ax.plot(I_like_hist, label='I_like_train')
ax.plot(val_I_like_hist, label='I_like_val')
ax.axhline(I_like, color='C2', linestyle=':', label='I_like_test')
ax.axhline(I_pred, color='C3', linestyle=':', label='I_pred_test')
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('bits')
ax.set_title('training hisotry')
ax.set_ylim([0, I_pred*1.2])

# Tighten bounds on figure
fig.tight_layout(w_pad=3)
fig.savefig('mpsa_ge_training.png')
plt.show()
