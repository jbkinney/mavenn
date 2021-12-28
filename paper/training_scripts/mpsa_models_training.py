"""
This script is for training MPSA models for the paper
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
Ammar Tareen, Mahdi Kooshkbaghi, Anna Posfai, 
William T. Ireland, David M. McCandlish, Justin B. Kinney
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

# Import MAVE-NN local
import sys

sys.path.insert(0, "../../")
import mavenn

# Fix the seed, default seed is seed=1234
mavenn.src.utils.set_seed()

# Get the date to append to the saved model name
today = datetime.now()
date_str = today.strftime("%Y.%m.%d.%Hh.%Mm")

# Get the models dictionary from the json input file.
input_file = open("mpsa_models_param.json")
models_dict = json.load(input_file)


def main(args):

    # Set dataset name
    dataset_name = "mpsa"

    # Set type of GP map
    model_type = args.model_type

    # Get parameters dict
    params_dict = models_dict[model_type]

    # Read dataset from the paper dataset instead of MAVENN dataset
    data_df = pd.read_csv(f"../datasets/{dataset_name}_data.csv.gz")

    # Get and report sequence length
    L = len(data_df.loc[0, "x"])
    print(f"Sequence length: {L:d} ")

    # Preview dataset
    print("data_df:")

    # Split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)

    # Preview trainval_df
    print("trainval_df:")

    # For blackbox model the json file can not have
    # the tuples for the hidden_layer_sizes.
    # here we convert the "(tuple)" to (tuple) for hidden_layer_size
    if model_type == "blackbox":
        hidden_layer_sizes = eval(
            params_dict["model_params"]["gpmap_kwargs"]["hidden_layer_sizes"]
        )
        params_dict["model_params"]["gpmap_kwargs"][
            "hidden_layer_sizes"
        ] = hidden_layer_sizes

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
    model_name = f"../models/mpsa_{model_type}_ge_{date_str}"
    model.save(model_name)

    # simulate 20 datasets for pairwise model
    if model_type=='pairwise':
        num_models = 20
        sim_models = model.sample_plausible_models(data_df=data_df,
                                                   num_models=num_models,
                                                   initialize_from_fit_model=True)
        # save simulated models
        for i in range(num_models):
            model_name = f"../models/mpsa_{model_type}_ge_model_{i}_{date_str}"
            sim_models[i].save(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splicing MPSA Models")
    parser.add_argument(
        "-m","--model_type", default="additive", type=str,
        help="Types of G-P maps: additive, neighbor, pairwise, blackbox",
    )
    args = parser.parse_args()

    main(args)
