import mavenn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng

GER = mavenn.load(mavenn.__path__[0] +'/examples/models/gaussian_GB1_model')

print('GE noise model type:', GER.ge_noise_model_type)

# equalate g(phi) for continuous phi
phi_range = np.linspace(-9,2.3,1000)
y_hat_GE = GER.phi_to_yhat(phi_range)

# Get quantiles
qs = GER.yhat_to_yq(y_hat_GE, q=np.array([0.16,0.84]))

# get test data
GB1_test_data = pd.read_csv(mavenn.__path__[0] +'/examples/datafiles/gb1/GB1_test_data.csv',index_col=[0])


# subsample test data set to display demo quickly
rng = default_rng()
numbers = rng.choice(len(GB1_test_data)-1, size=20000, replace=False)

GB1_test_data = GB1_test_data.loc[numbers].copy()

# put test seqs/predictions in convenient variables as np arrays
x_test = GB1_test_data['x_test'].values
y_test = GB1_test_data['y_test'].values

# predictions
y_hat = GER.x_to_yhat(x_test)

# evaluate phi for sequences
phi = GER.x_to_phi(x_test)

fig = plt.figure(figsize=(8,8))

gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])

ax1.scatter(phi,y_test, s=0.5,alpha=0.5)
ax1.plot(phi_range,GER.phi_to_yhat(phi_range))
ax1.plot(phi_range,GER.phi_to_yhat(phi_range),lw=2,label='$\hat{y}$',alpha=1.0,color='black')

for q_index in range(qs.shape[1]):
    ax1.plot(phi_range,qs[:,q_index].ravel(),color='orange',lw=2,alpha=0.85,label='$\hat{y} \pm \sigma(\hat{y})$')

ax1.set_ylabel('Observations')
ax1.set_xlabel('Latent phenotype ($\phi$)')
ax1.set_title(GER.ge_noise_model_type+' Likelihood')

ax2 = fig.add_subplot(gs[0, 1])

ax2.scatter(y_hat,y_test,s=0.5,alpha=0.25)
ax2.set_ylabel('Observations')
ax2.set_xlabel('Predictions ($\hat{y}$)')

ax3 = fig.add_subplot(gs[1, :])

# display heatmap
theta = pd.DataFrame(GER.get_gpmap_parameters()['value'][1:].values.reshape(55,20), columns=GER.model.characters)
WT_sequence_GB1 = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
mavenn.heatmap(theta, wt_seq=WT_sequence_GB1, cmap='PiYG', ax=ax3)

plt.tight_layout()
plt.show()