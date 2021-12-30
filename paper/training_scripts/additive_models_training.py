"""
This script is for training following additive models for the paper
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
Ammar Tareen, Mahdi Kooshkbaghi, Anna Posfai, 
William T. Ireland, David M. McCandlish, Justin B. Kinney

1. TDP43
2. ABeta
3. GB1
    3.1 GB1 with N=500 double mutant subsample
    3.2 GB1 with N=5000 double mutant subsample
    3.3 GB1 with N=50000 double mutant subsample
"""
# Standard imports
import numpy as np
import argparse
import pandas as pd
import warnings
import os
from datetime import datetime
import json

# Turn off Tensorflow GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Import Local MAVENN
import sys

sys.path.insert(0, "../../")
import mavenn

# Fix the seed, default is seed=1234
mavenn.src.utils.set_seed()

# Fix rng for random generator in gb1 subsampling
rng = np.random.default_rng(1234)

# Get the date to append to the saved model name
today = datetime.now()
date_str = today.strftime("%Y.%m.%d.%Hh.%Mm")

# Get the models dictionary from the json input file.
input_file = open("additive_models_param.json")
models_dict = json.load(input_file)


def main(args):

    # Set dataset name
    dataset_name = args.dataset_name

    # Get parameters dict
    params_dict = models_dict[dataset_name]

    data_file = params_dict['dataset_name']

    # Read dataset from the paper dataset instead of MAVENN dataset
    data_df = pd.read_csv(f"../datasets/{data_file}_data.csv.gz")

    # Get and report sequence length
    L = len(data_df.loc[0, "x"])
    print(f"Sequence length: {L:d} ")

    # Preview dataset
    print("data_df:")

    # Split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)

    # Preview trainval_df
    print("trainval_df:")

    # Define model
    model = mavenn.Model(L=L, **params_dict["model_params"])

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df["y"],
        validation_flags=trainval_df["validation"],
    )

    # Train model
    model.fit(verbose=False, **params_dict["fit_params"])

    # Compute variational information on test data
    I_var, dI_var = model.I_variational(x=test_df["x"], y=test_df["y"])
    print(f"test_I_var: {I_var:.3f} +- {dI_var:.3f} bits")

    # Compute predictive information on test data
    I_pred, dI_pred = model.I_predictive(x=test_df["x"], y=test_df["y"])
    print(f"test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits")

    # Save model to file
    model_name = f"../models/{dataset_name}_ge_additive_{date_str}"
    model.save(model_name)

    if dataset_name=='gb1':
        print('\nSubsampling GB1 dataset')
        # The gb1 dataset contains both single and double mutants.
        # extrace the single and double mutants from dataset
        gb1_single_df = data_df[data_df['dist']==1].reset_index(drop=True).copy()
        gb1_double_df = data_df[data_df['dist']==2].reset_index(drop=True).copy()
        len_double = len(gb1_double_df)
        print(f'gb1 dataset contains {len_double} double mutants')

        # Choose the subsample size from the double mutants
        subsample_sizes=[500, 5000, 50000]
        for size in subsample_sizes:
            idx = rng.choice(len_double,
                             size=size, 
                             replace=False)
            # Create the selected double mutant gb1 with size `size` 
            gb1_s_double_df = gb1_double_df.loc[idx].reset_index(drop=True).copy()
            # The subsample data_df contains gb1_s_double_df and gb1_single_df
            subsampled_df = gb1_single_df[['set','y','x']].append(gb1_s_double_df[['set','y','x']],
                                                                  ignore_index=True)

            # Split subsampled dataset
            train_df, test_df = mavenn.split_dataset(subsampled_df)
            # Get sequence length
            L = len(data_df['x'][0])

            # Define model
            sub_model = mavenn.Model(regression_type='GE',
                                     L=L,
                                     alphabet='protein',
                                     gpmap_type='additive',                     
                                     ge_noise_model_type='Gaussian',
                                     ge_heteroskedasticity_order=2)
            # Set training data
            sub_model.set_data(x=train_df["x"],
                               y=train_df["y"],
                               validation_flags=train_df["validation"],
                               shuffle=True)
            # Fit model to data
            print(f'\nTraining gb1 subsample with N={size}')
            sub_model.fit(learning_rate=5e-5,
                          epochs=10,
                          batch_size=50,
                          early_stopping=True,
                          early_stopping_patience=15,
                          linear_initialization=True,
                          verbose=False)
            sub_model_name = f"../models/{dataset_name}_ge_additive_sub_{size}_{date_str}"
            sub_model.save(sub_model_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein DMS Additive GP-maps")
    parser.add_argument(
        "-d", "--dataset_name", default="amyloid", type=str, help="Dataset Name"
    )
    args = parser.parse_args()
    main(args)