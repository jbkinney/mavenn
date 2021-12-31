import pandas as pd
import matplotlib.pyplot as plt
from epistasis.models import EpistasisLinearRegression
from epistasis.pyplot import plot_coefs

from epistasis.models.nonlinear import EpistasisPowerTransform
from epistasis.pyplot.nonlinear import plot_power_transform
from epistasis.pyplot.nonlinear import plot_scale

from sklearn.model_selection import train_test_split
import numpy as np

from gpmap import GenotypePhenotypeMap
MPSA_WT = 'CAGGUAAGU'

import sys
path_to_mavenn = '/Users/tareen/Desktop/Research_Projects/2020_mavenn_github/mavenn_git_ssh/'
sys.path.insert(0, path_to_mavenn)

# Load mavenn
import mavenn

# Download mpsa dataset from master branch
url = 'https://github.com/jbkinney/mavenn/blob/master/mavenn/examples/datasets/mpsa_data.csv.gz?raw=true'

data_df = pd.read_csv(url,  
                      compression='gzip',
                      index_col=[0])

data_df = data_df.reset_index().copy()

# Separate test from data_df
ix_test = data_df['set']=='test'
test_df = data_df[ix_test].reset_index(drop=True)
print(f'test N: {len(test_df):,}')

# Remove test data from data_df
data_df = data_df[~ix_test].reset_index(drop=True)


x_train = data_df['x'].values
y_train = data_df['y'].values

x_test = test_df['x'].values
y_test = test_df['y'].values

gpm = GenotypePhenotypeMap(MPSA_WT,x_train,y_train)

# Fit Power transform
model = EpistasisPowerTransform(lmbda=1, A=0, B=0)
model.add_gpm(gpm)
model.fit()

fig, ax = plt.subplots(figsize=(2, 2))
# plot_power_transform(model, cmap='plasma', ax=ax, yerr=0.6, s=5, alpha=0.5)
    
plot_scale(model, ax=ax, s=0.25, alpha=0.4, color='gray', cmap=None, linecolor='orange')
ax.set_ylabel('Observations', fontsize=7.5)
ax.set_xlabel('Latent phenotype ($\phi$)', fontsize=7.5)
plt.tick_params(labelsize=7.5)
ax.set_title('Power law', fontsize=7.5)
fig.savefig('21.12.30.power_law/power_law.png', bbox_inches='tight', dpi=600)

plt.figure(figsize=(2, 2))
plt.scatter(model.predict(x_test), y_test, s=0.25, alpha=0.4)
plt.xlabel('Predictions (test)', fontsize=7.5)
#Rsq = np.corrcoef(model.predict(x_test), y_test)[0][1] ** 2
plt.ylabel('Observations (test)', fontsize=7.5)
plt.tick_params(labelsize=7.5)
#plt.title('$R^2=$' + str(Rsq)[0:5], fontsize=7.5)
plt.tick_params(labelsize=7.5)
plt.savefig('21.12.30.power_law/power_law_y_test_y_pred_test.png', bbox_inches='tight', dpi=600)

#np.savetxt('power_transform_model_predictions.txt', model.predict(x_test))
    
np.savetxt('21.12.30.power_law/y_hat_on_test_SH_powerlaw.txt', model.predict(x_test))

y_add_test = np.loadtxt('phi_SH.txt')
#np.savetxt('results_2020.10.23/y_test.txt', y_test)
xx = np.linspace(min(y_add_test), max(y_add_test), 100)
yy = model.minimizer.predict(xx)
np.savetxt('21.12.30.power_law/y_add_range_SH.txt', xx)
np.savetxt('21.12.30.power_law/y_obs_range_SH.txt', yy)
print('All done')