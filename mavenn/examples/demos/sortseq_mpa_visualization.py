'''
demo: sortseq_mpa_visualization

Illustrates an additive G-P map trained using MPA regression
on the rnap-wt dataset of Kinney et al., 2010. Runs fast.
'''

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Import MAVE-NN and Logomaker
import mavenn
import logomaker

# Load model
model_file = mavenn.__path__[0] +'/examples/models/rnap-wt'
model = mavenn.load(model_file)

# Get G-P map parameters in matrix form
theta_df = model.get_additive_parameters(out_format="matrix")

# Create grid in phi space
phi_lim = [-5, 3]
phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)

# Create array of allowable y values
Y = model.model.Y    # Y = number of bins
y_lim = [-.5, Y-.5]
y_all = range(Y)

# Compute matrix of p(y|phi) values
measurement_process = model.p_of_y_given_phi(y_all, phi_grid)

# Create figure with two panels
fig, axs = plt.subplots(1,2,figsize=[12,4])

# Left panel: draw logo using logomaker
ax = axs[0]
logo = logomaker.Logo(theta_df, ax=ax)
ax.set_ylabel(r'parameter value ($\theta_{l:c}$)')
ax.set_xlabel(r'position ($l$)')
ax.set_title('G-P map parameters')

# Right panel: draw measurement process as heatmap
ax = axs[1]
im = ax.imshow(measurement_process,
               cmap='Greens',
               extent=phi_lim+y_lim,
               vmin=0,
               origin='lower',
               aspect="auto")
ax.set_yticks(y_all)
ax.set_ylabel('bin number (y)')
ax.set_xlabel('latent phenotype ($\phi$)')
ax.set_title('mesurement process')
cb = plt.colorbar(im)
cb.set_label('probaility  $p(y|\phi)$', rotation=-90, va="bottom")

# Fix up plot
fig.tight_layout(w_pad=3)
