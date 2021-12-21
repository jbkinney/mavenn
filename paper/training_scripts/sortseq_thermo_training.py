"""
This script is for thermodynamic model for sortseq data for the paper
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
Ammar Tareen, Mahdi Kooshkbaghi, Anna Posfai, 
William T. Ireland, David M. McCandlish, Justin B. Kinney
"""

# Standard imports
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import warnings
import os
from datetime import datetime
import json
import argparse
import matplotlib.pyplot as plt

# Turn off Tensorflow GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Standard TensorFlow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant

# Import MAVE-NN local
import sys

sys.path.insert(0, "../../")
import mavenn

# Import base class
from mavenn.src.layers.gpmap import GPMapLayer
from mavenn.src.utils import _x_to_mat

# Fix the seed, default seed is seed=1234
mavenn.src.utils.set_seed()

# Get the date to append to the saved model name
today = datetime.now()
date_str = today.strftime("%Y.%m.%d.%Hh.%Mm")

# Get the models dictionary from the json input file.
input_file = open("sortseq_thermo_param.json")
models_dict = json.load(input_file)

# Define custom G-P map layer


class sortseqGPMapLayer(GPMapLayer):
    """Represents an thermodynamic model of transcription
    regulation in E. Coli at the lac promoter, which
    contains binding sites for RNAP and CRP.
    """

    def __init__(
        self, TF_start, TF_end, RNAP_start, RNAP_end, regularizer, *args, **kwargs
    ):
        """Construct layer instance."""

        # set attributes
        self.TF_start = TF_start
        self.TF_end = TF_end
        self.RNAP_start = RNAP_start
        self.RNAP_end = RNAP_end
        self.C = kwargs["C"]
        self.regularizer = tf.keras.regularizers.L2(regularizer)

        # form helpful variables
        self.L_TF = TF_end - TF_start
        self.L_RNAP = RNAP_end - RNAP_start

        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""

        # define bias/chemical potential weight for crp
        self.mu_TF = self.add_weight(
            name="mu_TF",
            shape=(1,),
            initializer=Constant(1.0),
            trainable=True,
            regularizer=self.regularizer,
        )

        # define bias/chemical potential weight for rnap
        self.mu_RNAP = self.add_weight(
            name="mu_RNAP",
            shape=(1,),
            initializer=Constant(1.0),
            trainable=True,
            regularizer=self.regularizer,
        )

        # Define theta_TF_lc parameters
        theta_TF_lc_shape = (1, self.L_TF, self.C)

        theta_TF_lc_init = np.random.randn(*theta_TF_lc_shape) / np.sqrt(self.L_TF)
        self.theta_TF_lc = self.add_weight(
            name="theta_TF_lc",
            shape=theta_TF_lc_shape,
            # initializer=Constant(theta_TF_lc_init),
            trainable=True,
            regularizer=self.regularizer,
        )

        # Define theta_rnap_lc parameters
        theta_RNAP_lc_shape = (1, self.L_RNAP, self.C)

        theta_RNAP_lc_init = np.random.randn(*theta_RNAP_lc_shape) / np.sqrt(
            self.L_RNAP
        )
        self.theta_RNAP_lc = self.add_weight(
            name="theta_RNAP_lc",
            shape=theta_RNAP_lc_shape,
            # initializer=Constant(theta_RNAP_lc_init),
            trainable=True,
            regularizer=self.regularizer,
        )

        # define interaction term. Not sure if this needs regularization
        self.interaction = self.add_weight(
            name="interaction",
            shape=(1,),
            initializer=Constant(0),
            trainable=True,
            regularizer=tf.keras.regularizers.L2(0),
        )

        # define tsat term. Not sure if this needs regularization
        self.tsat = self.add_weight(
            name="tsat",
            shape=(1,),
            initializer=Constant(1.0),
            trainable=True,
            regularizer=tf.keras.regularizers.L2(0),
        )

        # Call superclass build
        super().build(input_shape)

    def call(self, x_lc):
        """Process layer input and return output.

        x_lc: (tensor)
            Input tensor that represents one-hot encoded
            sequence values.
        """

        # extract locations of binding sites from entire lac-promoter sequence.
        x_TF_lc = x_lc[:, self.C * self.TF_start : self.C * self.TF_end]
        x_RNAP_lc = x_lc[:, self.C * self.RNAP_start : self.C * self.RNAP_end]

        # reshape according to crp and rnap lengths.
        x_TF_lc = tf.reshape(x_TF_lc, [-1, self.L_TF, self.C])
        x_RNAP_lc = tf.reshape(x_RNAP_lc, [-1, self.L_RNAP, self.C])

        # compute delta G for crp
        phi_TF = self.mu_TF + tf.reshape(
            K.sum(self.theta_TF_lc * x_TF_lc, axis=[1, 2]), shape=[-1, 1]
        )

        # compute delta G for rnap
        phi_RNAP = self.mu_RNAP + tf.reshape(
            K.sum(self.theta_RNAP_lc * x_RNAP_lc, axis=[1, 2]), shape=[-1, 1]
        )

        # compute rate of transcription
        t = (
            (self.tsat)
            * (K.exp(-phi_RNAP) + K.exp(-phi_TF - phi_RNAP - self.interaction))
            / (
                1
                + K.exp(-phi_TF)
                + K.exp(-phi_RNAP)
                + K.exp(-phi_TF - phi_RNAP - self.interaction)
            )
        )

        # return rate of transcription
        return t


def simulation(trained_model, params_dict, lr, ep, n_sim=2, n_seq=51000):
    """
    Simulate the dataset
    trained_model: trained model
    param_dict: model parameters set in the json file
    lr: learning rate
    ep: number of epochs
    n_sim: number of simulation
    n_seq: number of sequences to simulate

    """

    for model_number in range(n_sim):

        sim_df_new = trained_model.simulate_dataset(n_seq)
        y_cols_sim = sim_df_new.columns[1:-1]
        # Separate test from data_df
        ix_test = sim_df_new["set"] == "test"
        test_df = sim_df_new[ix_test].reset_index(drop=True)
        # Remove test data from data_df
        sim_df_new = sim_df_new[~ix_test].reset_index(drop=True)
        # Create model
        sim_model = mavenn.Model(
            custom_gpmap=sortseqGPMapLayer,
            **params_dict["model_params"],
            gpmap_kwargs=params_dict["gpmap_kwargs"],
        )

        # Set training data
        sim_model.set_data(
            x=sim_df_new["x"],
            y=sim_df_new[y_cols_sim],
            validation_flags=(sim_df_new["set"] == "validation"),
            shuffle=True,
        )

        # Fit model to data
        sim_model.fit(learning_rate=lr, epochs=ep, **params_dict["fit_params"])

        # Save simulated model to file
        sim_model_name = f"../models/sortseq_sim_{model_number}_lr_{lr}_e_{ep}_{date_str}"
        sim_model.save(sim_model_name)


def main(args):

    # Set dataset name
    dataset_name = "sortseq"

    # Set type of GP map
    model_type = "custom"

    # Set learning rate
    learning_rate = args.learning_rate

    # Set number of epochs
    epochs = args.epochs

    # Get parameters dict
    params_dict = models_dict[model_type]

    # Read dataset from the paper dataset instead of MAVENN dataset
    data_df = pd.read_csv(f"../datasets/{dataset_name}_data.csv.gz")
    print(data_df)

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
    # Create model instance
    model = mavenn.Model(
        custom_gpmap=sortseqGPMapLayer,
        **params_dict["model_params"],
        gpmap_kwargs=params_dict["gpmap_kwargs"],
    )

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df[[col for col in trainval_df if col.startswith("ct")]],
        validation_flags=trainval_df["validation"],
    )

    # Train model
    model.fit(learning_rate=learning_rate, epochs=epochs, **params_dict["fit_params"])

    # Save trained model to file
    model_name = (
        f"../models/sortseq_{model_type}_lr_{learning_rate}_e_{epochs}_{date_str}"
    )
    model.save(model_name)

    # simulate new dataset with trained model
    simulation(
        trained_model=model,
        params_dict=params_dict,
        lr=learning_rate,
        ep=epochs,
        n_sim=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sortseq Thermodynamic Model")
    parser.add_argument(
        "-e", "--epochs", default=1000, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=1e-3, type=float, help="Number of epochs"
    )
    args = parser.parse_args()
    main(args)
