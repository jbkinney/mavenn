"""
This script is for training three state thermodynamic model for the paper
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
input_file = open("gb1_thermo_param.json")
models_dict = json.load(input_file)

# Define custom G-P map layer


class OtwinowskiGPMapLayer(GPMapLayer):
    """
    A G-P map representing the thermodynamic model described by
    Otwinowski (2018).
    """

    def __init__(self, *args, **kwargs):
        """Construct layer instance."""

        # Call superclass constructor
        # Sets self.L, self.C, and self.regularizer
        super().__init__(*args, **kwargs)

        # Initialize constant parameter for folding energy
        self.theta_f_0 = self.add_weight(name='theta_f_0',
                                         shape=(1,),
                                         trainable=True,
                                         regularizer=self.regularizer)

        # Initialize constant parameter for binding energy
        self.theta_b_0 = self.add_weight(name='theta_b_0',
                                         shape=(1,),
                                         trainable=True,
                                         regularizer=self.regularizer)

        # Initialize additive parameter for folding energy
        self.theta_f_lc = self.add_weight(name='theta_f_lc',
                                          shape=(1, self.L, self.C),
                                          trainable=True,
                                          regularizer=self.regularizer)

        # Initialize additive parameter for binding energy
        self.theta_b_lc = self.add_weight(name='theta_b_lc',
                                          shape=(1, self.L, self.C),
                                          trainable=True,
                                          regularizer=self.regularizer)

    def call(self, x_lc):
        """Compute phi given x."""

        # 1kT = 0.582 kcal/mol at room temperature
        kT = 0.582

        # Reshape input to samples x length x characters
        x_lc = tf.reshape(x_lc, [-1, self.L, self.C])

        # Compute Delta G for binding
        Delta_G_b = self.theta_b_0 + \
            tf.reshape(K.sum(self.theta_b_lc * x_lc, axis=[1, 2]),
                       shape=[-1, 1])

        # Compute Delta G for folding
        Delta_G_f = self.theta_f_0 + \
            tf.reshape(K.sum(self.theta_f_lc * x_lc, axis=[1, 2]),
                       shape=[-1, 1])

        # Compute and return fraction folded and bound
        Z = 1+K.exp(-Delta_G_f/kT)+K.exp(-(Delta_G_f+Delta_G_b)/kT)
        p_bf = (K.exp(-(Delta_G_f+Delta_G_b)/kT))/Z
        phi = p_bf  # K.log(p_bf)/np.log(2)
        return phi


def main(args):

    # Set dataset name
    dataset_name = "gb1"

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
    # Making full dataset available for training.
    # Change all the set values to 'training' for this dataset
    data_df['set'] = 'training'
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
    # Order the alphabet to match Otwinowski (2018)
    alphabet = np.array(list('KRHEDNQTSCGAVLIMPYFW'))

    # Create model instance
    model = mavenn.Model(alphabet=alphabet,
                         custom_gpmap=OtwinowskiGPMapLayer,
                         **params_dict["model_params"],
                         gpmap_kwargs=params_dict["gpmap_kwargs"])

    # Set training data
    model.set_data(
        x=trainval_df["x"],
        y=trainval_df["y"],
        validation_flags=trainval_df["validation"],
    )

    # Train model
    model.fit(learning_rate=learning_rate,
              epochs=epochs,
              **params_dict["fit_params"])

    # Save model to file
    model_name = f"../models/gb1_{model_type}_lr_{learning_rate}_e_{epochs}_ge_full_{date_str}"
    model.save(model_name)

    # Retrieve G-P map parameter dict and view dict keys
    theta_dict = model.layer_gpmap.get_params()
    theta_dict.keys()
    # Get the wild-type GB1 sequence
    wt_seq = model.x_stats['consensus_seq']

    # Convert this to a one-hot encoded matrix of size LxC
    x_lc_wt = _x_to_mat(wt_seq, model.alphabet)

    # Subtract wild-type character value from parameters at each position
    ddG_b_mat_mavenn = theta_dict['theta_b_lc'] - \
        np.sum(x_lc_wt*theta_dict['theta_b_lc'], axis=1)[:, np.newaxis]
    ddG_f_mat_mavenn = theta_dict['theta_f_lc'] - \
        np.sum(x_lc_wt*theta_dict['theta_f_lc'], axis=1)[:, np.newaxis]

    # Load Otwinowski parameters form file
    dG_b_otwinowski_df = pd.read_csv('../../mavenn/examples/datasets/raw/otwinowski_gb_data.csv.gz',
                                     index_col=[0]).T.reset_index(drop=True)[model.alphabet]
    dG_f_otwinowski_df = pd.read_csv('../../mavenn/examples/datasets/raw/otwinowski_gf_data.csv.gz',
                                     index_col=[0]).T.reset_index(drop=True)[model.alphabet]

    # Compute ddG matrices for Otwinowski
    ddG_b_mat_otwinowski = dG_b_otwinowski_df.values - \
        np.sum(x_lc_wt*dG_b_otwinowski_df.values, axis=1)[:, np.newaxis]
    ddG_f_mat_otwinowski = dG_f_otwinowski_df.values - \
        np.sum(x_lc_wt*dG_f_otwinowski_df.values, axis=1)[:, np.newaxis]

    # Load Nisthal data
    nisthal_df = mavenn.load_example_dataset('nisthal')
    nisthal_df.set_index('x', inplace=True)

    # Get Nisthal folding energies relative to WT
    dG_f_nisthal = nisthal_df['y']
    dG_f_wt_nisthal = dG_f_nisthal[wt_seq]
    ddG_f_nisthal = dG_f_nisthal - dG_f_wt_nisthal

    # Get MAVE-NN folding energies relative to WT
    x_nisthal = nisthal_df.index.values
    x_nisthal_ohe = mavenn.src.utils.x_to_ohe(x=x_nisthal,
                                              alphabet=model.alphabet)
    ddG_f_vec = ddG_f_mat_mavenn.ravel().reshape([1, -1])
    ddG_f_mavenn = np.sum(ddG_f_vec*x_nisthal_ohe, axis=1)

    # Get Otwinowski folding energies relative to WT
    ddG_f_vec_otwinowski = ddG_f_mat_otwinowski.ravel().reshape([1, -1])
    ddG_f_otwinowski = np.sum(ddG_f_vec_otwinowski*x_nisthal_ohe, axis=1)

    # Define plotting routine
    def draw(ax, y, model_name):
        Rsq = np.corrcoef(ddG_f_nisthal, y)[0, 1]**2
        ax.scatter(ddG_f_nisthal, y, alpha=.2, label='data')
        ax.scatter(0, 0, label='WT sequence')
        xlim = [-3, 5]
        ax.set_xlim(xlim)
        ax.set_ylim([-4, 8])
        ax.plot(xlim, xlim, color='k', alpha=.5, label='diagonal')
        ax.set_xlabel(f'Nisthal $\Delta \Delta G_F$ (kcal/mol)')
        ax.set_ylabel(f'{model_name} $\Delta \Delta G_F$ (kcal/mol)')
        ax.set_title(f'$R^2$ = {Rsq:.3f}')
        ax.legend()

    # Make figure
    fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    draw(ax=axs[0],
         y=ddG_f_otwinowski,
         model_name='Otwinowski')
    draw(ax=axs[1],
         y=ddG_f_mavenn,
         model_name='MAVE-NN')

    fig.tight_layout(w_pad=5)
    plt.savefig(f'gb1_comp_{learning_rate}_{epochs}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GB1 Thermodynamic Model")
    parser.add_argument(
        "-e", "--epochs", default=1000, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=1e-3, type=float, help="Number of epochs"
    )
    args = parser.parse_args()
    main(args)