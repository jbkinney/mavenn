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
    'learning_rate': 5e-4,
    'epochs': 500,
    'batch_size': 200,
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
fig, ax = plt.subplots(1, 1)

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
    mp_GE = mavenn.measurement_process_layers.GlobalEpsitasisMP(K=20)
    # Define Model
    model = mavenn.Model2(gpmap=gpmap, mp_list=[mp_GE])
    # Set training data
    model.set_data(x=data_df['x'],
                   y_list=[data_df['y'].values.reshape(-1, 1)],
                   validation_flags=(data_df['set'] == 'validation'),
                   shuffle=False)

    model.fit(**default_fit_kwargs)

    # Plot I_var_train, the variational information on training data as a function of epoch
    ax.plot(model.history['I_var'], color=line_colors[c],
            label=f'train Ivar {g}')
    # Plot I_var_val, the variational information on validation data as a function of epoch
    ax.plot(model.history['val_I_var'], '--',
            color=line_colors[c], label=f'val I_var {g}')

ax.legend(frameon=True, fontsize=8, loc=4)
ax.set_xlabel('epochs')
ax.set_ylabel('bits')
plt.title('MPSA GE modeling')

plt.tight_layout()
plt.savefig('gp_implementations.pdf')
plt.show()
