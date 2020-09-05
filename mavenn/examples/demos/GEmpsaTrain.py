'''
This demo trains a neighbor mpsa model
'''

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

# load mavenn's GE model
gpmap_type = 'neighbor'
ge_noise_model_type = 'Gaussian'
GER = mavenn.Model(regression_type='GE',
                   x=x_train,
                   y=y_train,
                   gpmap_type=gpmap_type,
                   ge_noise_model_type=ge_noise_model_type,
                   ge_nonlinearity_monotonic=True,
                   ge_nonlinearity_hidden_nodes=20,
                   alphabet='dna',
                   ge_heteroskedasticity_order=0,
                   theta_regularization=0.01,
                   eta_regularization=0.01)

GER.fit(epochs=20,
        learning_rate=0.005,
        early_stopping=False,
        early_stopping_patience=5,
        validation_split=0.1,
        verbose=1)

loss_history = GER.model.history

yhat = GER.x_to_yhat(x_test)
phi = GER.x_to_phi(x_test)

phi_range = np.linspace(min(phi),max(phi),1000)

model = GER.get_nn()

# predictions
yhat = GER.x_to_yhat(x_test)

# evaluate phi for sequences
phi = GER.x_to_phi(x_test)

# equalate g(phi) for continuous phi
phi_range = np.linspace(min(phi),max(phi),1000)
y_hat_GE = GER.phi_to_yhat(phi_range)

# noise model that is used to get eta parameters
qs = GER.yhat_to_yq(y_hat_GE,q=np.array([0.16,0.84]))

fig, ax = plt.subplots(1,3,figsize=(12,4))

ax[0].plot(loss_history.history['loss'], color='blue')
ax[0].plot(loss_history.history['val_loss'], color='orange')
ax[0].set_title(GER.gpmap_type+' model loss', fontsize=12)
ax[0].set_ylabel('loss', fontsize=12)
ax[0].set_xlabel('epoch', fontsize=12)
ax[0].legend(['train', 'validation'])

Rsq = np.corrcoef(yhat.ravel(),y_test)[0][1]**2
ax[1].scatter(yhat,y_test,s=5,alpha=0.4)
ax[1].set_xlabel('Predictions (test)')
ax[1].set_ylabel('Observations (test)')
ax[1].set_title('$R^2$: '+str(Rsq)[0:5])

ax[2].plot(phi_range,GER.phi_to_yhat(phi_range))
ax[2].scatter(phi,y_test,s=0.25, alpha=0.4, label='Observations')
ax[2].plot(phi_range,GER.phi_to_yhat(phi_range),lw=2,label='$\hat{y}$',alpha=1.0,color='black')

for q_index in range(qs.shape[1]):
    ax[2].plot(phi_range,qs[:,q_index].ravel(),color='orange',lw=2,alpha=0.85,label='$\hat{y} \pm \sigma(\hat{y})$')

ax[2].set_ylabel('Observations')
ax[2].set_xlabel('Latent phenotype ($\phi$)')
ax[2].set_title(GER.ge_noise_model_type+' Likelihood')

plt.tight_layout()
plt.show()