import mavenn
import numpy as np
import pandas as pd
import logomaker
import seaborn as sns
import matplotlib.pyplot as plt


MPAR = mavenn.load(mavenn.__path__[0] +'/examples/models/full-wt')

theta_df = pd.DataFrame(MPAR.get_gpmap_parameters()['value'].values[1:].reshape(39,4),columns=['A','C','G','T'])

phi_range = np.linspace(-8,3,500)
p_of_all_y_given_phi = MPAR.na_p_of_all_y_given_phi(phi_range)

fig, ax = plt.subplots(1,2,figsize=(8,3))

theta_df.index = np.arange(-39,0)
logomaker.Logo(theta_df,center_values=False,
               font_name='Arial Rounded MT Bold',
               ax=ax[0])

ax[0].set_xlabel('Position')
ax[0].set_ylabel('$\\theta$, (full-wt)')


ax = sns.heatmap(p_of_all_y_given_phi.T,cmap='Greens',ax=ax[1])
ax.invert_yaxis()

ax.set_ylabel('Bin number')
ax.set_xlabel('Latent Phenotype ($\phi$)')
ax.set_xticks(([0,int(len(phi_range)/2),len(phi_range)-2]), minor=False)
middle_tick = str(phi_range[int(len(phi_range)/2)])
ax.set_xticklabels(([str(phi_range[0])[0:5],middle_tick[0:2],str(phi_range[len(phi_range)-1])[0:5]]), minor=False)
plt.tight_layout()
plt.show()