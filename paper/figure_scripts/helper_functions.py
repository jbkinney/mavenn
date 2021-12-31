import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob


def my_rsquared(x,y, bootstrap_samples=1000):
    """Returns R^2 and SE thereof based on bootstrap resampling"""
    r2 = pearsonr(x,y)[0]**2
    N = len(x)
    assert len(x)==len(y), f'len(x)={len(x)} and len(y)={len(y)} are not the same.'
    r2s = np.zeros(bootstrap_samples)
    for i in range(bootstrap_samples):
        ix = np.random.choice(a=bootstrap_samples, size=bootstrap_samples, replace=True)
        r2s[i] = pearsonr(x[ix],y[ix])[0]**2
    dr2 = np.std(r2s)
    return r2, dr2


def save_fig_with_date_stamp(fig,
                             fig_name,
                             dpi=400,
                             facecolor='white',
                             **kwargs):
    """Saves figure with date-stamped name"""
    for file_name in glob.glob(f'png/{fig_name}*'):
        os.remove(file_name)
    from datetime import datetime
    time = datetime.now().strftime("%Y.%m.%d.%Hh.%Mm.%Ss")
    fig_file = f'png/{fig_name}_ipynb_{time}.png'
    fig.savefig(fig_file, dpi=dpi, facecolor=facecolor, **kwargs)
    print(f'Figure saved figure to {fig_file}.')

def set_xticks(ax, L, pos_start, pos_spacing):
    x_ticks = np.array(range(-pos_start % pos_spacing,
                             L,
                             pos_spacing)).astype(int)
    x_ticklabels = pos_start + x_ticks
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)