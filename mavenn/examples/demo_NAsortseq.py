import mavenn
import numpy as np
import logomaker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mavenn.src.utils import na_plots_for_mavenn_demo

# load data
sequences = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt', dtype='str')
bin_number = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/bin_numbers.txt')

# load mavenn's NA model
NA_model = mavenn.NoiseAgnosticModel(X=sequences, y=bin_number)
model = NA_model.define_model()
NA_model.compile_model(lr=0.005)
history = NA_model.fit(epochs=50, use_early_stopping=True, early_stopping_patience=10, verbose=1)

# evaluate the inferred noise model for a given input range
phi_range = np.linspace(-20,20,1000)
noise_model = NA_model.noise_model(input_range=phi_range)

# plot results using helper function
na_plots_for_mavenn_demo(history, model, noise_model, phi_range)
