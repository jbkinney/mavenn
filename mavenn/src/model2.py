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
# TODO: need to finish updating various gpmap implementations (e.g., pairwise custom)

# Tensorflow imports
# note model import has to be imported using an alias to avoid
# conflict with the Model class.
from tensorflow.keras.models import Model as TF_Functional_Model
from tensorflow.keras.layers import Input, Lambda, Concatenate

from mavenn.src.layers.measurement_process_layers \
    import GaussianNoiseModelLayer, \
           EmpiricalGaussianNoiseModelLayer, \
           CauchyNoiseModelLayer, \
           SkewedTNoiseModelLayer, \
           MPAMeasurementProcessLayer

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

        # need to add one additional node for targets, which will be used
        # calcalate likelihood (i.e., using x and y)
        # Note that if additional input features are to be added (e.g., shape),
        # the following code will have to be updated.
        number_input_layer_nodes = number_x_nodes + 1

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + 1],
            output_shape=((1,)),
            trainable=False, name='Labels_input')(inputTensor)

        # assign phi to gpmap input into constructor
        phi = self.gpmap(sequence_input)

        # TODO: will need to fix this in accordance with pseudocode above.
        # e.g., may be using a map datastructure
        yhat = self.mp_list[0](phi)

        yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
            [yhat, labels_input])

        # TODO: need to figure out how to assign noise model based on mp_list.
        # for GE, I will probably have to refactor the noise noise
        # to be included in the in the Global epistasis layer.
        # Using hardcoded Gaussian noise for now
        noise_model_layer = GaussianNoiseModelLayer(info_for_layers_dict={},
                                                    polynomial_order=2,
                                                    eta_regularization=1e-5)

        outputTensor = noise_model_layer(yhat_y_concat)

        model = TF_Functional_Model(inputTensor, outputTensor)
        self.model = model

        return model
