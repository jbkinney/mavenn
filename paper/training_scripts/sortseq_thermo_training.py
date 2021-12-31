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

# Import MAVE-NN
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
    """
    Represents a four-stage thermodynamic model
    containing the states:
    1. free DNA 
    2. CPR-DNA binding
    3. RNAP-DNA binding
    4. CPR and RNAP both bounded to DNA and interact
    """

    def __init__(self,
                 tf_start,
                 tf_end,
                 rnap_start,
                 rnap_end,
                 *args, **kwargs):
        """Construct layer instance."""


        # Call superclass
        super().__init__(*args, **kwargs)
        
        # set attributes
        self.tf_start = tf_start            # transcription factor starting position
        self.tf_end = tf_end                 # transcription factor ending position
        self.L_tf = tf_end - tf_start        # length of transcription factor
        self.rnap_start = rnap_start         # RNAP starting position
        self.rnap_end = rnap_end             # RNAP ending position
        self.L_rnap = rnap_end - rnap_start  # length of RNAP
        self.C = kwargs["C"]

        # define bias/chemical potential weight for TF/CRP energy
        self.theta_tf_0 = self.add_weight(name='theta_tf_0',
                                          shape=(1,),
                                          initializer=Constant(1.),
                                          trainable=True,
                                          regularizer=self.regularizer)

        # define bias/chemical potential weight for rnap energy
        self.theta_rnap_0 = self.add_weight(name='theta_rnap_0',
                                            shape=(1,),
                                            initializer=Constant(1.),
                                            trainable=True,
                                            regularizer=self.regularizer)

        # initialize the theta_tf
        theta_tf_shape = (1, self.L_tf, self.C)
        theta_tf_init = np.random.randn(*theta_tf_shape)/np.sqrt(self.L_tf)
        
        # define the weights of the layer corresponds to theta_tf
        self.theta_tf = self.add_weight(name='theta_tf',
                                        shape=theta_tf_shape,
                                        initializer=Constant(theta_tf_init),
                                        trainable=True,
                                        regularizer=self.regularizer)

        # define theta_rnap parameters
        theta_rnap_shape = (1, self.L_rnap, self.C)
        theta_rnap_init = np.random.randn(*theta_rnap_shape)/np.sqrt(self.L_rnap)
        
        # define the weights of the layer corresponds to theta_rnap
        self.theta_rnap = self.add_weight(name='theta_rnap',
                                          shape=theta_rnap_shape,
                                          initializer=Constant(theta_rnap_init),
                                          trainable=True,
                                          regularizer=self.regularizer)

        # define trainable real number G_I, representing interaction Gibbs energy
        self.theta_dG_I = self.add_weight(name='theta_dG_I',
                                   shape=(1,),
                                   initializer=Constant(-4),
                                   trainable=True,
                                   regularizer=self.regularizer)


    def call(self, x):
        """Process layer input and return output.

        x: (tensor)
            Input tensor that represents one-hot encoded 
            sequence values. 
        """
        
        # 1kT = 0.616 kcal/mol at body temperature
        kT = 0.616

        # extract locations of binding sites from entire lac-promoter sequence.
        # for transcription factor and rnap
        x_tf = x[:, self.C * self.tf_start:self.C * self.tf_end]
        x_rnap = x[:, self.C * self.rnap_start: self.C * self.rnap_end]

        # reshape according to tf and rnap lengths.
        x_tf = tf.reshape(x_tf, [-1, self.L_tf, self.C])
        x_rnap = tf.reshape(x_rnap, [-1, self.L_rnap, self.C])

        # compute delta G for crp binding
        G_C = self.theta_tf_0 + \
            tf.reshape(K.sum(self.theta_tf * x_tf, axis=[1, 2]),
                       shape=[-1, 1])

        # compute delta G for rnap binding
        G_R = self.theta_rnap_0 + \
            tf.reshape(K.sum(self.theta_rnap * x_rnap, axis=[1, 2]),
                       shape=[-1, 1])
        
        G_I = self.theta_dG_I

        # compute phi
        numerator_of_rate = K.exp(-G_R/kT) + K.exp(-(G_C+G_R+G_I)/kT)
        denom_of_rate = 1.0 + K.exp(-G_C/kT) + K.exp(-G_R/kT) + K.exp(-(G_C+G_R+G_I)/kT)
        phi = numerator_of_rate/denom_of_rate

        return phi


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
    model_name = (f"../models/sortseq_thermodynamic_mpa_{date_str}")
    model.save(model_name)

    # simulate new dataset with trained model
    num_models = args.num_models
    sim_models = model.bootstrap(data_df=data_df,
                                 num_models=num_models,
                                 initialize_from_self=True)
    # save simulated models
    for i in range(num_models):
        model_name = (f"../models/sortseq_thermodynamic_mpa_model_{i}_{date_str}")
        sim_models[i].save(model_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sortseq Thermodynamic Model")
    parser.add_argument(
        "-e", "--epochs", default=2000, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=1e-4, type=float, help="Learning Rate"
    )

    parser.add_argument(
        "-ns", "--num_models", default=20, type=int, help="number of simulation models"
        )


    args = parser.parse_args()
    main(args)