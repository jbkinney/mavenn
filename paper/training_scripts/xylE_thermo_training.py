"""
This script is for thermodynamic model for xylE for the paper
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
Ammar Tareen, Mahdi Kooshkbaghi, Anna Posfai, 
William T. Ireland, David M. McCandlish, Justin B. Kinney
"""

# Standard imports
import numpy as np
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
input_file = open("xylE_thermo_param.json")
models_dict = json.load(input_file)

# Define custom G-P map layer


class xylEGPMapLayer(GPMapLayer):

    """Represents an thermodynamic model of transcription
    regulation in E. Coli at the xylE promoter, which
    contains binding sites for RNAP, CRP, and xylR.
    """

    def __init__(
        self,
        CRP_start,
        CRP_end,
        xylR_start,
        xylR_end,
        RNAP_start,
        RNAP_end,
        regularizer,
        *args,
        **kwargs,
    ):
        """Construct layer instance."""

        # set attributes
        self.CRP_start = CRP_start
        self.CRP_end = CRP_end

        self.xylR_start = xylR_start
        self.xylR_end = xylR_end

        self.RNAP_start = RNAP_start
        self.RNAP_end = RNAP_end

        self.C = kwargs["C"]
        self.regularizer = tf.keras.regularizers.L2(regularizer)

        # form helpful variables
        self.L_CRP = CRP_end - CRP_start
        self.L_RNAP = RNAP_end - RNAP_start
        self.L_xylR = xylR_end - xylR_start

        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""

        # 1 define bias/chemical potential weight for crp
        self.mu_CRP = self.add_weight(
            name="mu_CRP",
            shape=(1,),
            initializer=Constant(0.0),
            trainable=True,
            regularizer=self.regularizer,
        )

        # 2 define bias/chemical potential weight for xylR
        self.mu_xylR = self.add_weight(
            name="mu_xylR",
            shape=(1,),
            initializer=Constant(0.0),
            trainable=True,
            regularizer=self.regularizer,
        )

        # 3 define bias/chemical potential weight for rnap
        self.mu_RNAP = self.add_weight(
            name="mu_RNAP",
            shape=(1,),
            initializer=Constant(0.0),
            trainable=True,
            regularizer=self.regularizer,
        )

        # 4 Define theta_CRP_lc parameters
        theta_CRP_lc_shape = (1, self.L_CRP, self.C)

        theta_CRP_lc_init = np.random.randn(*theta_CRP_lc_shape) / np.sqrt(self.L_CRP)
        self.theta_CRP_lc = self.add_weight(
            name="theta_CRP_lc",
            shape=theta_CRP_lc_shape,
            trainable=True,
            regularizer=self.regularizer,
        )

        # 5 Define theta_xylR_lc parameters
        theta_xylR_lc_shape = (1, self.L_xylR, self.C)

        theta_xylR_lc_init = np.random.randn(*theta_xylR_lc_shape) / np.sqrt(
            self.L_xylR
        )
        self.theta_xylR_lc = self.add_weight(
            name="theta_xylR_lc",
            shape=theta_xylR_lc_shape,
            trainable=True,
            regularizer=self.regularizer,
        )

        # 6 Define theta_rnap_lc parameters
        theta_RNAP_lc_shape = (1, self.L_RNAP, self.C)

        theta_RNAP_lc_init = np.random.randn(*theta_RNAP_lc_shape) / np.sqrt(
            self.L_RNAP
        )
        self.theta_RNAP_lc = self.add_weight(
            name="theta_RNAP_lc",
            shape=theta_RNAP_lc_shape,
            trainable=True,
            regularizer=self.regularizer,
        )

        # 7 define interaction term between CRP and xylR
        self.I_cx = self.add_weight(
            name="I_cx", shape=(1,), initializer=Constant(-1), trainable=True
        )

        # 8 define interaction term between RNAP and xylR.
        self.I_rx = self.add_weight(
            name="I_rx",
            shape=(1,),
            initializer=Constant(-1),
            trainable=True,
            regularizer=self.regularizer,
        )

        # 10 define tsat term.
        self.tsat = self.add_weight(
            name="tsat", shape=(1,), initializer=Constant(1.0), trainable=True
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
        x_CRP_lc = x_lc[:, self.C * self.CRP_start : self.C * self.CRP_end]
        x_xylR_lc = x_lc[:, self.C * self.xylR_start : self.C * self.xylR_end]
        x_RNAP_lc = x_lc[:, self.C * self.RNAP_start : self.C * self.RNAP_end]

        # reshape according to crp and rnap lengths.
        x_CRP_lc = tf.reshape(x_CRP_lc, [-1, self.L_CRP, self.C])
        x_xylR_lc = tf.reshape(x_xylR_lc, [-1, self.L_xylR, self.C])
        x_RNAP_lc = tf.reshape(x_RNAP_lc, [-1, self.L_RNAP, self.C])

        # compute delta G for crp
        phi_CRP = self.mu_CRP + tf.reshape(
            K.sum(self.theta_CRP_lc * x_CRP_lc, axis=[1, 2]), shape=[-1, 1]
        )

        # compute delta G for LacI
        phi_xylR = self.mu_xylR + tf.reshape(
            K.sum(self.theta_xylR_lc * x_xylR_lc, axis=[1, 2]), shape=[-1, 1]
        )

        # compute delta G for rnap
        phi_RNAP = self.mu_RNAP + tf.reshape(
            K.sum(self.theta_RNAP_lc * x_RNAP_lc, axis=[1, 2]), shape=[-1, 1]
        )

        # partition function
        Z = (
            1
            + K.exp(-phi_CRP)
            + K.exp(-phi_xylR)
            + K.exp(-phi_RNAP)
            + K.exp(-phi_xylR - phi_RNAP - self.I_rx)
            + K.exp(-phi_CRP - phi_RNAP)
            + K.exp(-phi_CRP - phi_xylR - phi_RNAP - self.I_cx - self.I_rx)
            + K.exp(-phi_CRP - phi_xylR - self.I_cx)
        )

        transcription_states = (
            K.exp(-phi_RNAP)
            + K.exp(-phi_xylR - phi_RNAP - self.I_rx)
            + K.exp(-phi_CRP - phi_RNAP)
            + K.exp(-phi_CRP - phi_xylR - phi_RNAP - self.I_cx - self.I_rx)
        )

        # compute rate of transcription
        t = (self.tsat) * (transcription_states) / Z

        # return rate of transcription
        return t


def main(args):

    # Set dataset name
    dataset_name = "xylE"

    # Set type of GP map
    model_type = "custom"

    # Set learning rate
    learning_rate = args.learning_rate

    # Set number of epochs
    epochs = args.epochs

    # Get parameters dict
    params_dict = models_dict[model_type]

    # Read dataset from the paper dataset instead of MAVENN dataset
    # data_df = pd.read_csv(f"../datasets/{dataset_name}_data.csv.gz")
    data_df = mavenn.load_example_dataset("xylE")
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
        custom_gpmap=xylEGPMapLayer,
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

    # Save model to file
    model_name = f"../models/xylE_{model_type}_lr_{learning_rate}_e_{epochs}_{date_str}"
    model.save(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xylE Thermodynamic Model")
    parser.add_argument(
        "-e", "--epochs", default=200, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=3e-4, type=float, help="Number of epochs"
    )
    args = parser.parse_args()
    main(args)
