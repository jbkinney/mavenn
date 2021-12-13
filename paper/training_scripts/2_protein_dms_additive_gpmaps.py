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


def data_set_dict():
    """
    This function will define the dataset name and
    the associated parametes used for training the models
    in the paper. This is the only function we need to
    tune if we are not happy with the models.
    """
    models_dict = {}
    # Parametes for the amyloid beta trained in the paper
    models_dict["amyloid"] = {}
    models_dict["amyloid"]["dataset_name"] = "amyloid"
    models_dict["amyloid"]["alphabet"] = "protein*"
    models_dict["amyloid"]["gpmap_type"] = "additive"
    models_dict["amyloid"]["ge_noise_model_type"] = "SkewedT"
    models_dict["amyloid"]["ge_heteroskedasticity_order"] = 2
    models_dict["amyloid"]["learning_rate"] = 1e-3
    models_dict["amyloid"]["epochs"] = 500
    models_dict["amyloid"]["batch_size"] = 64
    models_dict["amyloid"]["early_stopping_patience"] = 25

    return models_dict


def loading_dataset_and_params(dataset_name):
    models_dict = data_set_dict()
    alphabet = models_dict[dataset_name]["alphabet"]
    gp_map_type = models_dict[dataset_name]["gpmap_type"]
    ge_noise_model_type = models_dict[dataset_name]["ge_noise_model_type"]
    ge_heteroskedasticity_order = models_dict[dataset_name][
        "ge_heteroskedasticity_order"
    ]
    learning_rate = models_dict[dataset_name]["learning_rate"]
    epochs = models_dict[dataset_name]["epochs"]
    batch_size = models_dict[dataset_name]["batch_size"]
    early_stopping_patience = models_dict[dataset_name]["early_stopping_patience"]

    # Load datset
    print(f"Loading dataset and parameters for '{dataset_name}' ")
    data_df = mavenn.load_example_dataset(dataset_name)

    return (
        data_df,
        alphabet,
        gp_map_type,
        ge_noise_model_type,
        ge_heteroskedasticity_order,
        learning_rate,
        epochs,
        batch_size,
        early_stopping_patience,
    )


def main(args):

    dataset_name = args.dataset_name
    (
        data_df,
        alphabet,
        gp_map_type,
        ge_noise_model_type,
        ge_heteroskedasticity_order,
        learning_rate,
        epochs,
        batch_size,
        early_stopping_patience,
    ) = loading_dataset_and_params(dataset_name)

    # Get and report sequence length
    L = len(data_df.loc[0, "x"])
    print(f"Sequence length: {L:d} amino acids (+ stops)")

    # Preview dataset
    print("data_df:")

    # Split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)

    # Preview trainval_df
    print("trainval_df:")

    # Define model
    model = mavenn.Model(
        L=L,
        alphabet=alphabet,
        gpmap_type=gp_map_type,
        regression_type="GE",
        ge_noise_model_type=ge_noise_model_type,
        ge_heteroskedasticity_order=ge_heteroskedasticity_order,
    )

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df["y"],
        validation_flags=trainval_df["validation"],
    )

    # Train model
    model.fit(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        verbose=False,
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
