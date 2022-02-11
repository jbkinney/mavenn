"""
2022.02.04
----------
model2.py: Defines the Model() class, which represents all MAVE-NN2 models.
Unlike version, this model class contains new features such as custom
measurement processes, new measurement processes such as tite-seq,
multi-latent phenotype models, and more.

Pseudocode showing some of the updated workflow and new features

# Define GP map (sets dimensionality of latent phenotype phi)
gpmap = ThermodynamicGPMap(...)

# Define measurement processes (specify dimensions of phi and form of y)
mp_ge = GEMeasurementProcess(...)
mp_mpa = MPAMeasurementProcess(...)

# Define model
model = Model(gpmap = gpmap,
              mplist = [mp_ge, mp_mpa])

# Set data
model.set_data(x = x,
               y_list = [y_ge, y_mpa],
               validation_flags = validation_flags)

# Fit model
model.fit(...)
"""

# TODO: need to develop model2's define model more according to TODOS in define_model().
# TODO: need to define helper function's for model in keynote (e.g. mp_ge.phi_to_yhat ... )
# TODO: need to finish implementation of mp_list (using Ammardev branch)
# TODO: need to finish updating various gpmap implementations (e.g., pairwise, custom, and refactor x to x_ohe)

# Tensorflow imports
# note model import has to be imported using an alias to avoid
# conflict with the Model class.
from tensorflow.keras.models import Model as TF_Functional_Model
from tensorflow.keras.layers import Input, Lambda, Concatenate

from mavenn.src.layers.input_layer import InputLayer
from mavenn.src.layers.measurement_process_layers \
    import GlobalEpsitasisMP, \
           GaussianNoiseModelLayer, \
           EmpiricalGaussianNoiseModelLayer, \
           CauchyNoiseModelLayer, \
           SkewedTNoiseModelLayer, \
           DiscreteAgnosticMP

class Model:

    """
     Represents a MAVE-NN (version 2) model, which includes a genotype-phenotype (G-P) map
     as well as a list of measurement processes.

     Parameters
     ----------
     gpmap: (MAVE-NN gpmap)
         MAVE-NN's Genotype-phenotype object.

     mp_list: (list)
        List of measurement processes. 

    """

    def __init__(self,
                 gpmap,
                 mp_list):

        # set attributes required for defining a model
        self.gpmap = gpmap

        self.L = self.gpmap.L
        self.C = self.gpmap.C

        self.mp_list = mp_list

        # define model
        self.model = self.define_model()

    def define_model(self):

        """
        Method that defines the neural network
        model using the gpmap and mp_list passed to
        the constructor
        """

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

        # Get input layer tensor, the sequence input, and the labels input
        input_tensor, sequence_input, labels_input = InputLayer(number_x_nodes).get_input_layer()

        # assign phi to gpmap input into constructor
        phi = self.gpmap(sequence_input)

        # get list of measurement processes
        measurement_processes_list = self.mp_list

        # loop over measurement processes and create a list of output tensors
        # that phi will be connected to. Currently output tensors get assigned
        # to the list, they are invoked slightly separate based on whether they
        # output a regression value (yhat) or counts (like discrete agnostic MPA)
        # but this is subject to change

        output_tensor_list = []
        # index to keep track of names of different y_yhat_likelihood layers
        idx_y_and_yhat_ll = 0
        for current_mp in measurement_processes_list:

            # if measurement process object has yhat attribute
            # note prediction in GE is yhat, but MPA it would be phi
            if hasattr(current_mp, 'yhat'):

                yhat = current_mp.yhat(phi)

                # concatenate y_hat and y to pass into likelihood computation layer
                prediction_y_concat = Concatenate(name=f'yhat_and_y_to_ll_{idx_y_and_yhat_ll}')(
                    [yhat, labels_input])

                idx_y_and_yhat_ll += 1
                output_tensor = current_mp.mp_layer(prediction_y_concat)
                output_tensor_list.append(output_tensor)
            else:

                # concatenate phi and counts to pass into likelihood computation layer
                prediction_y_concat = Concatenate()([phi, labels_input])
                output_tensor = current_mp(prediction_y_concat)
                output_tensor_list.append(output_tensor)

        model = TF_Functional_Model(input_tensor, output_tensor_list)

        self.model = model

        return model
