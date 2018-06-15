# Simulate library example

import mpathic as mpa

# reate a library of random mutants from an initial wildtype sequence and mutation rate
sim_library = mpa.SimulateLibrary(wtseq="TAATGTGAGTTAGCTCACTCAT", mutrate=0.24)
sim_library.output_df.head()

# Load dataset and model dataframes
dataset_df = mpa.io.load_dataset('sort_seq_data.txt')
model_df = mpa.io.load_model('crp_model.txt')

# Simulate a Sort-Seq experiment example
sim_sort = mpa.SimulateSort(df=dataset_df,mp=model_df)
sim_sort.output_df.head()