import mavenn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load data
MPSA_data = pd.read_csv(mavenn.__path__[0] +'/examples/datafiles/mpsa/brca2_lib1_rep1.csv')

X = MPSA_data['ss'].values
y = MPSA_data['log_psi'].values
dy = MPSA_data['dlog_psi'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

GER_additive = mavenn.load(mavenn.__path__[0] +'/examples/models/skewT_mpsa_model_additive')
GER_pairwise = mavenn.load(mavenn.__path__[0] +'/examples/models/skewT_mpsa_model_pairwise')

# Make predictions and compute latent phenotype values for each model
# Additive:
yhat_additive = GER_additive.x_to_yhat(x_test)

# evaluate phi for sequences
phi_additive = GER_additive.x_to_phi(x_test)

# equalate g(phi) for continuous phi
phi_range_additive = np.linspace(min(phi_additive),max(phi_additive),1000)
y_hat_GE_additive = GER_additive.phi_to_yhat(phi_range_additive)

# noise model that is used to get eta parameters
qs_additive = GER_additive.yhat_to_yq(y_hat_GE_additive,q=np.array([0.16,0.84]))

# Pairwise:
yhat_pairwise = GER_pairwise.x_to_yhat(x_test)

# evaluate phi for sequences
phi_pairwise = GER_pairwise.x_to_phi(x_test)

# equalate g(phi) for continuous phi
phi_range_pairwise = np.linspace(min(phi_pairwise),max(phi_pairwise),1000)
y_hat_GE_pairwise = GER_pairwise.phi_to_yhat(phi_range_pairwise)

# noise model that is used to get eta parameters
qs_pairwise = GER_pairwise.yhat_to_yq(y_hat_GE_pairwise,q=np.array([0.16,0.84]))

# show plots
fig, ax = plt.subplots(2,2,figsize=(8,8))

Rsq = np.corrcoef(yhat_additive.ravel(),y_test)[0][1]**2
ax[0,0].scatter(yhat_additive,y_test,s=5,alpha=0.4)
ax[0,0].set_xlabel('Predictions (test)')
ax[0,0].set_ylabel('Observations (test)')
ax[0,0].set_title('$R^2$: '+str(Rsq)[0:5]+' (additive)')

ax[0,1].plot(phi_range_additive,GER_additive.phi_to_yhat(phi_range_additive))
ax[0,1].scatter(phi_additive,y_test,s=0.25, alpha=0.4, label='Observations')
ax[0,1].plot(phi_range_additive,GER_additive.phi_to_yhat(phi_range_additive),lw=2,label='$\hat{y}$',alpha=1.0,color='black')

for q_index in range(qs_additive.shape[1]):
    ax[0,1].plot(phi_range_additive,qs_additive[:,q_index].ravel(),color='orange',lw=2,alpha=0.85,label='$\hat{y} \pm \sigma(\hat{y})$')

ax[0,1].set_ylabel('Observations')
ax[0,1].set_xlabel('Latent phenotype ($\phi$)')
ax[0,1].set_title(GER_additive.ge_noise_model_type+' Likelihood, (additive)')

Rsq = np.corrcoef(yhat_pairwise.ravel(),y_test)[0][1]**2
ax[1,0].scatter(yhat_pairwise,y_test,s=5,alpha=0.4)
ax[1,0].set_xlabel('Predictions (test)')
ax[1,0].set_ylabel('Observations (test)')
ax[1,0].set_title('$R^2$: '+str(Rsq)[0:5]+' (pairwise)')

ax[1,1].plot(phi_range_pairwise,GER_pairwise.phi_to_yhat(phi_range_pairwise))
ax[1,1].scatter(phi_pairwise,y_test,s=0.25, alpha=0.4, label='Observations')
ax[1,1].plot(phi_range_pairwise,GER_pairwise.phi_to_yhat(phi_range_pairwise),lw=2,label='$\hat{y}$',alpha=1.0,color='black')

for q_index in range(qs_pairwise.shape[1]):
    ax[1,1].plot(phi_range_pairwise,qs_pairwise[:,q_index].ravel(),color='orange',lw=2,alpha=0.85,label='$\hat{y} \pm \sigma(\hat{y})$')

ax[1,1].set_ylabel('Observations')
ax[1,1].set_xlabel('Latent phenotype ($\phi$)')
ax[1,1].set_title(GER_pairwise.ge_noise_model_type+' Likelihood, (pairwise)')


plt.tight_layout()
plt.show()