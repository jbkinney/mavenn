import mavenn
import numpy as np
from mavenn.src.utils import na_plots_for_mavenn_demo

# load data
sequences = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/rnap_sequences.txt', dtype='str')
bin_counts = np.loadtxt(mavenn.__path__[0]+'/examples/datafiles/sort_seq/full-wt/bin_counts.txt')

# load mavenn's NA model
NAR = mavenn.Model(regression_type='NA',
                   X=sequences,
                   y=bin_counts,
                   model_type='additive',
                   alphabet_dict='dna')

NAR.fit(epochs=200,
        use_early_stopping=True,
        early_stopping_patience=20,
        verbose=1)

loss_history =  NAR.model.return_loss()

# evaluate the inferred noise model for a given input range
phi_range = np.linspace(-20,20,1000)
noise_model = NAR.na_noisemodel(sequences,
                                input_range=phi_range)

# plot results using helper function
na_plots_for_mavenn_demo(loss_history, NAR.nn_model(), noise_model, phi_range)
