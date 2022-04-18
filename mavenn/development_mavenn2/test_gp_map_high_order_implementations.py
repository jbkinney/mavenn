import os
import warnings
# Ignore tensorflow CUDA backend warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
# Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# Path to local mavenn
sys.path.insert(0, '/home/mahdik/workspace/mavenn/')
import mavenn

dataset = 'mpsa'
# Load example data
data_df = mavenn.load_example_dataset(dataset)

# Split dataset
trainval_df, test_df = mavenn.split_dataset(data_df)

# Preview trainval_df
print('trainval_df:')
print(trainval_df)

# Length of the sequences
L = len(data_df['x'][0])
# Alphabet for RNA
alphabet = ['A', 'C', 'G', 'U']

# Default fitting kwargs
default_fit_kwargs = {
    'learning_rate': 1e-3,
    'epochs': 500,
    'batch_size': 50,
    'early_stopping': True,
    'early_stopping_patience': 30,
    'linear_initialization': False,
    'verbose': False
}


gpmap_cases = ['AdditiveGPMapLayer',
               'K1',
               'PairwiseGPMapLayer',
               'K2',
               'K3']

line_colors = ['k', 'r', 'g', 'b', 'darkorange', 'purple']
fig, axs = plt.subplots(1, len(gpmap_cases) + 1,
                        figsize=(3 * (len(gpmap_cases) + 1), 3))

# Loop over gp-map implementations
for c, g in enumerate(gpmap_cases):
    print(g)
    if g == 'AdditiveGPMapLayer':
        gpmap = mavenn.gpmap.AdditiveGPMapLayer(L=L, alphabet=alphabet)

    if g == 'K1':
        gpmap = mavenn.gpmap.KOrderGPMap(
            L=L, alphabet=alphabet, interaction_order=1)

    if g == 'PairwiseGPMapLayer':
        gpmap = mavenn.gpmap.PairwiseGPMapLayer(
            L=L, alphabet=alphabet, mask_type='pairwise')

    if g == 'K2':
        gpmap = mavenn.gpmap.KOrderGPMap(
            L=L, alphabet=alphabet, interaction_order=2)

    if g == 'K3':
        gpmap = mavenn.gpmap.KOrderGPMap(
            L=L, alphabet=alphabet, interaction_order=3)

    # Initialize measurement process
    mp_GE = mavenn.measurement_process_layers.GlobalEpsitasisMP(
        K=50, ge_noise_model_type='SkewedT', ge_heteroskedasticity_order=2)
    # Define Model
    model = mavenn.Model2(gpmap=gpmap, mp_list=[mp_GE])
    # Set training data
    model.set_data(x=trainval_df['x'],
                   y_list=[trainval_df['y'].values.reshape(-1, 1)],
                   validation_flags=trainval_df['validation'],
                   shuffle=False)

    model.fit(**default_fit_kwargs)
    # TODO: model.save is broken
    # Get test data y values
    y_test = test_df['y']
    # Compute phi on test data
    phi_test = model.gpmap.x_to_phi(test_df['x'])

    # Set phi lims and create a grid in phi space
    phi_lim = [min(phi_test) - .5, max(phi_test) + .5]
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000).astype('float32')

    yhat_grid = mp_GE.phi_to_yhat(phi_grid)

    ax = axs[c + 1]
    # Plot GE nonlinearity
    ax.scatter(phi_test, y_test, c='C10', s=5,
               alpha=0.1, zorder=-10, rasterized=True)
    ax.plot(phi_grid, yhat_grid, linewidth=3,
            color=line_colors[c], label=f'G-P model {g}')

    # Style plot
    ax = axs[0]
    # Plot I_var_train, the variational information on training data as a function of epoch
    ax.plot(model.history['I_var'], color=line_colors[c],
            label=f'train Ivar {g}')
    # Plot I_var_val, the variational information on validation data as a function of epoch
    ax.plot(model.history['val_I_var'], '--',
            color=line_colors[c], label=f'val I_var {g}')

fs = 8
for c in range(len(gpmap_cases)):
    ax = axs[c + 1]
    ax.set_xlabel('latent phenotype ($\phi$)', fontsize=fs)
    ax.set_ylim([-1.5, 4.5])
    ax.set_ylabel('measurement ($y$)', fontsize=fs)
    ax.legend(frameon=True, loc='lower right', fontsize=fs)
# Style plots
ax = axs[0]
# ax.legend(frameon=True, fontsize=fs, loc='lower right')
ax.set_xlabel('epochs', fontsize=fs)
ax.set_ylabel('bits', fontsize=fs)


plt.title('MPSA GE modeling')
plt.tight_layout()
plt.savefig('gp_implementations.pdf')
plt.show()
