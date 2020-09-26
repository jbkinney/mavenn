"""
run_demo: gb1_ge_evaluation

Illustrates an additive G-P map, trained on the GB1 data of
Olson et al., 2014, fit using GE regression with a
heteroskedastic Gaussian noise model. Runs fast.
"""

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MAVE-NN
import mavenn

# Load model
model = mavenn.load_example_model('gb1_ge_additive')

# Set wild-type sequence
gb1_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

# Get effects of all single-point mutations on phi
theta_dict = model.get_theta(gauge='user',
                             x_wt=gb1_seq)

# Load data as dataframe
data_file = mavenn.__path__[0] + '/examples/datasets/gb1/GB1_test_data.csv'
data_df = pd.read_csv(data_file, index_col=[0])

# Subsample test data, just to make plotting faster
N_test = len(data_df)
N_sample = 5000
ix = np.random.choice(N_test, size=N_sample, replace=False).astype(int)

# Extract data into np.arrays
x = data_df['x_test'].values[ix]
y = data_df['y_test'].values[ix]

# Compute phi and yhat values
phi = model.x_to_phi(x)
yhat = model.phi_to_yhat(phi)

# Create grid for plotting yhat and yqs
phi_lim = [-5, 2.5]
phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
yhat_grid = model.phi_to_yhat(phi_grid)
yqs_grid = model.yhat_to_yq(yhat_grid, q=[.16,.84])

# Create two panels
fig, axs = plt.subplots(1, 2, figsize=[12, 4])

# Left panel: draw heatmap illustrating 1pt mutation effects
ax = axs[0]
ax, cb = mavenn.heatmap(theta_dict['theta_lc'],
                        alphabet=theta_dict['alphabet'],
                        seq=gb1_seq,
                        cmap='PiYG',
                        ccenter=0,
                        ax=ax)
ax.set_xlabel('position ($l$)')
ax.set_ylabel('amino acid ($c$)')
cb.set_label('effect ($\Delta\phi$)', rotation=-90, va="bottom")
ax.set_title('mutation effects')

# Right panel: illustrate measurement process with GE curve
ax = axs[1]
ax.scatter(phi, y, color='C0', s=5, alpha=.2, label='test data')
ax.plot(phi_grid, yhat_grid, linewidth=2, color='C1',
        label='$\hat{y} = g(\phi)$')
ax.plot(phi_grid, yqs_grid[:, 0], linestyle='--', color='C1',
        label='68% CI')
ax.plot(phi_grid, yqs_grid[:, 1], linestyle='--', color='C1')
ax.set_xlim(phi_lim)
ax.set_xlabel('latent phenotype ($\phi$)')
ax.set_ylabel('measurement ($y$)')
ax.set_title('measurement process')
ax.legend()

# Fix up plot
fig.tight_layout(w_pad=3)
fig.savefig('gb1_ge_evaluation.png')
plt.show()
