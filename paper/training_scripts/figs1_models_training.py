"""
This script is for training models for the Fig S1 of the paper:
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
import time

# Turn off Tensorflow GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Import MAVENN
import mavenn

# Fix the seed, default is seed=1234
mavenn.src.utils.set_seed()

# Fix rng for random generator in gb1 subsampling
rng = np.random.default_rng(1234)

# Get the date to append to the saved model name
today = datetime.now()
date_str = today.strftime("%Y.%m.%d.%Hh.%Mm")

# Get the models dictionary from the json input file.
input_file = open("figs1_models_param.json")
models_dict = json.load(input_file)


def main(args):

    # Set dataset name
    dataset_name = 'gb1'

    # Get parameters dict
    params_dict = models_dict[dataset_name]

    # Read dataset from the paper dataset instead of MAVENN dataset
    data_df = pd.read_csv(f"../datasets/{dataset_name}_data.csv.gz")

    # Get and report sequence length
    L = len(data_df.loc[0, "x"])
    print(f"Sequence length: {L:d} ")

    # Preview dataset
    print("data_df:")

    # Split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)

    x_test = test_df['x'].values
    y_test = test_df['y'].values

    # Preview trainval_df
    print("trainval_df:")

    # Define model
    model = mavenn.Model(L=L, **params_dict["model_params"])

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df["y"],
        validation_flags=trainval_df["validation"],
        shuffle=True)

    # Train model
    model.fit(verbose=False, **params_dict["fit_params"])

    # Compute variational information on test data
    I_var, dI_var = model.I_variational(x=test_df["x"], y=test_df["y"])
    print(f"test_I_var: {I_var:.3f} +- {dI_var:.3f} bits")

    # Compute predictive information on test data
    I_pred, dI_pred = model.I_predictive(x=test_df["x"], y=test_df["y"])
    print(f"test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits")

    # Save model to file
    model_name = f"../models/fig_s1_models/{dataset_name}_ge_additive_{date_str}"
    model.save(model_name)
    
    # ground truth values
    phi_test = model.x_to_phi(x_test)
    yhat_test = model.x_to_yhat(x_test)

    # Bootstrap models
    print('Bootstrap Models Training')
    num_models = args.num_bootstrap
    gb1_boot_model = model.bootstrap(data_df=data_df, 
                                    num_models=num_models,
                                    initialize_from_self=True)
    # save bootstraped models
    for i in range(num_models):
        gb1_boot_model_name = f"../models/fig_s1_models/bootsraping_models/{dataset_name}_ge_additive_bootstrap_{i}_{date_str}"
        gb1_boot_model[i].save(gb1_boot_model_name)

    # timing models
    print('Timing Models Training')
    N_samples = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    for N in N_samples:
        # size here represents the number of double mutants
        # uniformly randomly pick some number of double mutants
        numbers = rng.choice(len(data_df), size=N, replace=False) 
        data_df_N = data_df.loc[numbers].reset_index(drop=True).copy()
        # simulate data from loaded model
        sim_df = model.simulate_dataset(template_df=data_df_N)
        time_models = []
        training_times = []

        for model_index in range(10):
            print(f'training model {model_index}')
            sim_model = mavenn.Model(L=L,
                         alphabet='protein',
                         gpmap_type='additive', 
                         regression_type='GE',
                         ge_noise_model_type='Gaussian',
                         ge_heteroskedasticity_order=2)

            # Set simulated training data
            sim_model.set_data(x=sim_df['x'],
                               y=sim_df['y'],
                               shuffle=True,
                               verbose=False)
    
            start_time = time.time()
            # Fit model to data
            sim_model.fit(learning_rate=.005,
                          epochs=1000,
                          batch_size=200,
                          early_stopping=True,
                          early_stopping_patience=30,
                          linear_initialization=True,
                          verbose=False)
    
            training_time = time.time() - start_time
    
            training_times.append(training_time)
            time_models.append(sim_model)
        model_Rsqs = []
        for model_index in range(len(time_models)):
            print(f' Model: {model_index}, $R^2$: {np.corrcoef(time_models[model_index].x_to_yhat(x_test),yhat_test)[0][1]**2}')
            model_Rsqs.append(np.corrcoef(time_models[model_index].x_to_yhat(x_test),yhat_test)[0][1]**2)
            time_model_name = f'../models/fig_s1_models/timing_models/gb1_N_{N}_model_{model_index}_{date_str}'
            time_models[model_index].save(time_model_name)
        # save timining and R2 for each N samples
        np.savetxt(f'../models/fig_s1_models/timing_models/gb1_N_{N}_training_times_{date_str}.txt',training_times)
        np.savetxt(f'../models/fig_s1_models/timing_models/gb1_N_{N}_model_Rsqs_{date_str}.txt',model_Rsqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GB1 Fig S1")
    parser.add_argument("-ns", "--num_bootstrap", default=2, 
                        type=int, help="Number of bootstraping")
    args = parser.parse_args()
    main(args)