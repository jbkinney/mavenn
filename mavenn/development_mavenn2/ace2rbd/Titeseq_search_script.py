#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, '/Users/ammar.tareen/Desktop/Research_projects/mavenn2/')
sys.path.insert(0, '/Users/tareen/Desktop/Research_Projects/2022_mavenn2_github/mavenn')

import mavenn
import logomaker
import seaborn as sns
import re
from sklearn.model_selection import train_test_split

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_df_full_lib_1 = pd.read_csv('../data/2022.05.15.ace2rbd/ace2rbd_raw_counts_lib1.csv')
y_cols = list(data_df_full_lib_1.columns[8:])
cols = y_cols.copy()
cols.insert(0,'x')
data_df_lib_1 = data_df_full_lib_1[cols].copy()

data_df_full_lib_2 = pd.read_csv('../data/2022.05.15.ace2rbd/ace2rbd_raw_counts_lib2.csv')
y_cols = list(data_df_full_lib_2.columns[8:])
cols = y_cols.copy()
cols.insert(0,'x')
data_df_lib_2 = data_df_full_lib_2[cols].copy()

data_df = pd.concat([data_df_lib_1,data_df_lib_2]).reset_index(drop=True).copy()

X = data_df['x']
y = data_df[y_cols]



# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# set positional argumnets for gpmap function
L = len(data_df['x'][0])
alphabet=['A', 'C', 'D', 'E', 'F',
          'G', 'H', 'I', 'K', 'L',
          'M', 'N', 'P', 'Q', 'R',
          'S', 'T', 'V', 'W', 'Y',]


Y = len(y_cols)
#Y = 8
#print(f'L={L}, Y={Y}')

gpmap = mavenn.gpmap.AdditiveGPMapLayer(L, alphabet)
N_y = np.sum(y,axis=0)

bounds = np.log(np.array([[1,180],[180,1400],[1400,10500],[10500,250000]]))
#bounds = np.log(np.array([[1,180],[180,1400],[1400,10500],[10500,200000]]))

bounds_df = pd.DataFrame(bounds,columns=['lower_bound','upper_bound'])
f_y_lower_bounds = bounds_df['lower_bound'].values
f_y_upper_bounds = bounds_df['upper_bound'].values
bounds_df

#np.arange(10**(-13),10**(-6),10**(-12))
cs = [0,10**(-13),10**(-12.5),
     10**(-12),10**(-11.5),
     10**(-11),10**(-10.5),
     10**(-10),10**(-9.5),
     10**(-9),10**(-8.5),
     10**(-8),10**(-7.5),
     10**(-7),10**(-6.5),
     10**(-6)]



def mu_of_phi(c, 
              a,
              phi,
              mu_neg):

    K_a_of_phi = 10**(phi)
    B = 10**(mu_neg)
    A = 10**(a)
    
    mu_of_phi = np.log10(A*(c*K_a_of_phi)/(1+c*K_a_of_phi)+B)
    
    return mu_of_phi


def _x_to_mat(x, alphabet):
    return (np.array(list(x))[:, np.newaxis] == alphabet[np.newaxis, :]).astype(float)


# In[5]:


#mu_neg = 1
#mu_pos = 9

# sigma_neg = 1.5
# sigma_pos = 1.5

def run_search(mu_pos, mu_neg, sigma):

    sigma_pos = sigma
    sigma_neg = sigma
    
    a = np.log10(10**(mu_pos) - 10**mu_neg)

    mp_list = [mavenn.measurement_process_layers.TiteSeqMP(N_y=N_y[4*mp_idx:4*mp_idx+4],
                                                             c=cs[mp_idx],
                                                             a=a,
                                                             Y=4,
                                                             mu_pos=mu_pos,
                                                             sigma_pos=sigma_pos,
                                                             mu_neg=mu_neg,
                                                             sigma_neg=sigma_neg,
                                                             f_y_lower_bounds=f_y_lower_bounds,
                                                             f_y_upper_bounds=f_y_upper_bounds,
                                                             info_for_layers_dict={'H_y_norm':0},
                                                             eta=1e-5,)
               for mp_idx in range(len(y_cols)//4)
              ]

    model = mavenn.Model2(gpmap=gpmap,
                          mp_list=mp_list)

    # Set training data
    model.set_data(x=x_train,
                   verbose=False,
                   y_list=[
                           y_train[y_cols[0:4]].values,
                           y_train[y_cols[4:8]].values,
                           y_train[y_cols[8:12]].values,
                           y_train[y_cols[12:16]].values,
                           y_train[y_cols[16:20]].values,
                           y_train[y_cols[20:24]].values,
                           y_train[y_cols[24:28]].values,
                           y_train[y_cols[28:32]].values,
                           y_train[y_cols[32:36]].values,
                           y_train[y_cols[36:40]].values,
                           y_train[y_cols[40:44]].values,
                           y_train[y_cols[44:48]].values,
                           y_train[y_cols[48:52]].values,
                           y_train[y_cols[52:56]].values,                   
                           y_train[y_cols[56:60]].values,                                      
                           y_train[y_cols[60:64]].values,                                                         
                          ],

                   shuffle=True)


    # Fit model to data
    model.fit(learning_rate=.0005,
              epochs=300,
              batch_size=400,
              try_tqdm = False,
              early_stopping=False,
              verbose=False,
              early_stopping_patience=10,
              linear_initialization=False)

    # Show training history
    #print('On test data:')
    # x_test = test_df['x'].values
    # y_test = test_df[y_cols].values


    I_var_hist = model.history['I_var']
    val_I_var_hist = model.history['val_I_var']

    fig, axs = plt.subplots(1,2,figsize=[8,4])

    ax = axs[0]
    ax.plot(I_var_hist, label='I_var_train')
    ax.plot(val_I_var_hist, label='I_var_val')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('bits')
    #ax.set_xscale('log')
    ax.set_title('training hisotry')

    ax = axs[1]
    ax.plot(model.history['loss'], label='loss_train')
    ax.plot(model.history['val_loss'], label='loss_val')
    # ax.set_xlim(100,300)
    # ax.set_ylim(0.7*1e6,0.8*1e6)
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_title('training hisotry')
    fig.tight_layout()

    phi = gpmap.x_to_phi(x_test)
    theta = np.squeeze(gpmap.weights[1].numpy())

    # Get G-P map parameters in matrix form


    # Create grid in phi space
    phi_lim = [-15, 15]
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)

    # Create array of allowable y values
    Y = mp_list[0].Y    # Y = number of bins
    y_lim = [-.5, Y-.5]
    y_all = range(Y)

    # Compute matrix of p(y|phi) values



    # Create figure with two panels
    fig, axs = plt.subplots(4,4,figsize=[16,16])

    mp_counter = 0
    for i in range(4):
        for j in range(4):

            measurement_process = mp_list[mp_counter].p_of_y_given_phi(y_all, phi_grid)
            # Right panel: draw measurement process as heatmap
            ax = axs[i,j]

            im = ax.imshow(measurement_process,
                           cmap='Greens',
                           extent=phi_lim+y_lim,
                           vmin=0,
                           origin='lower',
                           interpolation='nearest',
                           aspect="auto")
            ax.set_yticks(y_all)
            ax.set_ylabel('bin number (y)')
            ax.set_xlabel('latent phenotype ($\phi$)')
            ax.set_title(f'Titeseq MP-{mp_counter+1}, c = {cs[mp_counter]:.3e}')
            mp_counter+=1

    cb = plt.colorbar(im)
    cb.set_label('probability  $p(y|\phi)$', rotation=-90, va="bottom")
    fig.tight_layout()

    fig.savefig(f'pngs/ace2rbd_Titeseq_measurment_prcoess_mu_pos_{(mu_pos):.3f}_{(mu_neg):.3f}.png',dpi=300,bbox_inches='tight')


    bloom_single_mut_df = pd.read_csv('single_mut_effects_Bloom.txt')

    bloom_bind_df = pd.DataFrame(columns=alphabet)
    for site in range(1,201+1,):

        temp_df = bloom_single_mut_df[bloom_single_mut_df['site_RBD']==site].copy()
        #temp_df['bind_lib1']
        #expr_lib1
        bloom_bind_df.loc[site-1] = temp_df['bind_avg'].values[0:20]

    bloom_bind_df.head()

    bloom_single_mut_df = pd.read_csv('single_mut_effects_Bloom.txt')
    bloom_exp_df = pd.DataFrame(columns=alphabet)
    for site in range(1,201+1,):

        temp_df = bloom_single_mut_df[bloom_single_mut_df['site_RBD']==site].copy()
        temp_df['expr_lib1']
        bloom_exp_df.loc[site-1] = temp_df['expr_avg'].values[0:20]

    bloom_exp_df.head()    

    theta_df = pd.DataFrame(theta,columns=alphabet)

#    fig, axs = plt.subplots(1,1,figsize=[4,4])

    # Left panel: draw logo using logomaker
#    ax = axs
    Rsq = np.corrcoef(bloom_bind_df.fillna(0).values.ravel(), theta_df.fillna(0).values.ravel())[0,1]**2
#     ax.scatter(bloom_bind_df.fillna(0).values,theta_df.fillna(0).values,s=2,alpha=0.2,color='blue')
#     ax.set_xlabel('Sinlge mut effects - Bloom')
#     ax.set_ylabel('MAVE-NN titeseq GPMAP')
#     ax.set_title(f'$R^2 = {Rsq:.3f}$')
#     # ax.set_xticks(np.arange(0,201,5))
#     # ax.set_xticklabels([f'{x}' for x in range(331,531+1,5)])
#     plt.show()


    theta_lc = gpmap.get_theta(model,gauge='consensus')['theta_lc']
    theta_lc_consensus_df = pd.DataFrame(theta_lc,columns=alphabet)

#    plt.figure(figsize=(4,4))
    Rsq = np.corrcoef(bloom_bind_df.fillna(0).values.ravel(), theta_lc_consensus_df.fillna(0).values.ravel())[0,1]**2
#    plt.scatter(bloom_bind_df.values, theta_lc_consensus_df.values,s=2.5,alpha=0.2,color='blue')

    lims = [-5,1]
#     #plt.plot(lims,lims,'--',color='gray',zorder=-1)
#     plt.xlabel('Bloom single mut effects - Binding')
#     plt.title(f'Titeseq measurement process \n $R^2 = {Rsq:.3f}$')
#     plt.ylabel('MAVE-NN2, G-P map')
#     plt.tight_layout()
#     plt.savefig('mavenn_bloom_binding_comparison.png',dpi=200,bbox_inches='tight')

#     plt.figure(figsize=(4,4))
    Rsq = np.corrcoef(bloom_exp_df.fillna(0).values.ravel(), theta_lc_consensus_df.fillna(0).values.ravel())[0,1]**2
#     plt.scatter(bloom_exp_df.values, theta_lc_consensus_df.values,s=2.5,alpha=0.25,color='blue')

#     #lims = [-5,1]
#     #plt.plot(lims,lims,'--',color='gray',zorder=-1)
#     plt.xlabel('Bloom single mut effects - Expression')
#     plt.title(f'Titeseq measurement process \n $R^2 = {Rsq:.3f}$')
#     plt.ylabel('MAVE-NN2, G-P map')
#     plt.tight_layout()
#     plt.savefig('mavenn_bloom_expression_comparison.png',dpi=200,bbox_inches='tight')

    theta_lc_consensus_dm_df = pd.read_csv('discrete_monotonic_ace2rbd_binding.csv')

#     plt.figure(figsize=(4,4))
    Rsq = np.corrcoef(theta_lc_consensus_dm_df.fillna(0).values.ravel(), theta_lc_consensus_df.fillna(0).values.ravel())[0,1]**2
#     plt.scatter(theta_lc_consensus_dm_df.values, theta_lc_consensus_df.values,s=2.5,alpha=0.2,color='blue')

#     #lims = [-5,1]
#     #plt.plot(lims,lims,'--',color='gray',zorder=-1)
#     plt.xlabel('Discrete monotonic G-P map')
#     plt.title(f'Titeseq measurement vs discrete monotonic MP \n GP-map conparison $R^2 = {Rsq:.3f}$')
#     plt.ylabel('Titeseq G-P map')
#     plt.tight_layout()
#     plt.savefig('Titeseq_discrete_monotonic_binding_comparison.png',dpi=200,bbox_inches='tight')




    fig, ax = plt.subplots(figsize=(5,5))

    WT_seq = model.x_stats['consensus_seq']

    single_mutants = []

    for idx,WT_char in enumerate(WT_seq):
        #print(idx)
        for mutant in model.alphabet:
    #         if mutant==WT_seq[idx]:
    #             continue
    #         else:
            temp_WT_list = list(WT_seq)
            temp_WT_list[idx] = mutant
            single_mutant = ''.join(temp_WT_list)
            single_mutants.append(single_mutant)

    phi_single_mutants = gpmap.x_to_phi(single_mutants)
    mavenn2_Kd = pd.DataFrame(1/(10**phi_single_mutants.reshape(201,20)),columns=model.alphabet)        

    fig, ax = plt.subplots(figsize=(5,5))

#     ax.scatter(mavenn2_Kd.values.ravel(),10**bloom_bind_df.values.ravel()
#                 ,s=5,alpha=0.5,color='blue')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     # ax.set_ylabel('$\log_{10}(K_d)$ (Starr et al)',fontsize=14)
#     # ax.set_xlabel('$\log_{10}(K_d)$, MAVE-NN, ${{\\rm exp}_{10}(\phi)}^{-1}$',fontsize=14)

#     ax.set_ylabel('$(K_d)$ (Starr et al)',fontsize=14)
#     ax.set_xlabel('$(K_d)$, MAVE-NN, ${{\\rm exp}_{10}(\phi)}^{-1}$',fontsize=14)

    phi_WT = gpmap.x_to_phi(WT_seq)
    cs_ = np.arange(10**-13,10**-7,10**(-11))
    cs_course = np.arange(10**-12,10**-1,10**(-7))
    cs_fig_1  = np.array([10**(-13),10**(-12),10**(-11),10**(-10),10**(-9),10**(-8),10**(-7),10**(-6),10**(-3),10**(-1)])

    #for a in as_list:
    #ax.plot(cs_course,(mu_of_phi(c=cs_course,a=a,phi=phi_WT)),lw=3,zorder=10,label=f'$A(\phi) = {(10**a):.1e}$')    
    ax.plot(cs_fig_1,(mu_of_phi(c=cs_fig_1,a=a,phi=phi_WT,mu_neg=mu_neg)),'ro--',lw=3,zorder=10,label=f'$A(\phi) = {(10**a):.1e}$')    

    ax.set_xlabel('Concentration [M]',fontsize=12)
    ax.set_ylabel('$\mu(\phi = \phi_{\\rm WT})$',fontsize=15)

    Kd_WT = 1/(10**phi_WT)
    ax.axvline(Kd_WT,color='black',lw=3,label=f'$K_d (WT) = {(Kd_WT):2e}$')
    for c in cs:    
        ax.axvline(c,c='gray',zorder=-10,alpha=0.4)

    ax.axvline(c,c='gray',label=f'Conc. for each MP',zorder=-10,alpha=0.0)    

    leg = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)


    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(10**-14,)    
    #ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.savefig(f'pngs/mu_phi_WT_vs_conc_{mu_pos}_{mu_neg}_{sigma_pos}.png',dpi=300,bbox_inches='tight')


# In[6]:


import sys

#print(float(sys.argv[1]))
run_search(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]))


