"""
This script is for training the ABeta and TDP-43 models 
for the paper
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
Ammar Tareen, Mahdi Kooshkbaghi, Anna Posfai, 
William T. Ireland, David M. McCandlish, Justin B. Kinney
"""
# Standard imports
import numpy as np
import argparse
import pandas as pd
import random
import os
from datetime import date
# Fix all possible seeds
import tensorflow as tf
tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)
os.environ["PYTHONHASHSEED"] = str(1234)


# Import MAVE-NN local
import sys

sys.path.insert(0, "/home/mahdik/workspace/mavenn")
import mavenn

# Get the date to append to the saved model name
today = date.today()
date_str = today.strftime("%d_%m_%Y")


# This dict defines the dataset name and
# the associated parameters used for training the models
# in the paper.
models_dict = {}

# Parameters for the amyloid beta trained in the paper
models_dict["amyloid"] = {
    "dataset_name":"amyloid",
    "model_params":{
        "alphabet": "protein*",
        "gpmap_type": "additive",
        "regression_type": "GE",
        "ge_noise_model_type": "SkewedT",
        "ge_heteroskedasticity_order": 2
    },
    "fit_params":{
        "learning_rate": 1e-3,
        "epochs": 500,
        "batch_size": 64,
        "early_stopping_patience": 25,
        "early_stopping":True
    }
}

def main(args):

    # Set dataset name
    dataset_name = args.dataset_name

    # Get parameters dict
    params_dict = models_dict[dataset_name]

    # TODO: Change to pandas
    data_df = mavenn.load_example_dataset(dataset_name)

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
    model = mavenn.Model(
        L=L,
        **params_dict["model_params"]
    )

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df["y"],
        validation_flags=trainval_df["validation"],
    )

    # Train model
    model.fit(
        verbose=False,
        **params_dict["fit_params"]
    )

    # Compute variational information on test data
    I_var, dI_var = model.I_variational(x=test_df["x"], y=test_df["y"])
    print(f"test_I_var: {I_var:.3f} +- {dI_var:.3f} bits")

    # Compute predictive information on test data
    I_pred, dI_pred = model.I_predictive(x=test_df["x"], y=test_df["y"])
    print(f"test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits")

    # Save model to file
    model_name = f"{dataset_name}_additive_ge_{date_str}"
    model.save(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein DMS Additive GP-maps")
    parser.add_argument(
        "-d", "--dataset_name", default="amyloid", type=str, help="Dataset Name"
    )
    args = parser.parse_args()
    main(args)
