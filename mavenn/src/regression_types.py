"""regression_types.py: Specialized classes for GE and MPA regression."""

# Standard imports
import numpy as np
import numbers

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_alphabet
from mavenn.src.layers.gpmap \
    import AdditiveGPMapLayer, \
    PairwiseGPMapLayer, \
    MultilayerPerceptronGPMap, \
    Multi_AdditiveGPMapLayer,  \
    Multi_PairwiseGPMapLayer, \
    ThermodynamicGPMapLayer

from mavenn.src.layers.measurement_process_layers \
    import GlobalEpistasisLayer, \
        AffineLayer, \
        GaussianNoiseModelLayer, \
        CauchyNoiseModelLayer, \
        SkewedTNoiseModelLayer, \
        MPAMeasurementProcessLayer, \
        MultiMPAMeasurementProcessLayer, \
        MultiPhiGlobalEpistasisLayer


@handle_errors
class GlobalEpistasisModel:
    """
    Represents a global epistatsis model.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox']

    ge_nonlinearity_type: (str)
        Specifies the form of the GE nonlinearity. Options:
        "linear": An affine transformation from phi to yhat.
        "nonlinear": Allow and arbitrary nonlinear map from phi to yhat.

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 alphabet,
                 gpmap_type,
                 gpmap_kwargs,
                 ge_nonlinearity_monotonic,
                 ge_nonlinearity_type,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ge_nonlinearity_type = ge_nonlinearity_type

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that ge_nonlinearity_type valid
        allowed_types = ['linear', 'nonlinear']
        check(self.ge_nonlinearity_type in allowed_types,
              f'gpmap_type = {self.gpmap_type};'
              f'must be in {allowed_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):
        """
        Establish model architecture.

        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        Parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        Returns
        -------
        model: (tf.model)
            TensorFlow model that can be compiled and subsequently fit to data.
        """
        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT'},
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

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


        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = AdditiveGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = PairwiseGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        elif self.gpmap_type == 'blackbox':
            self.x_to_phi_layer = MultilayerPerceptronGPMap(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                **self.gpmap_kwargs)
        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Make global epistasis layer
        if self.ge_nonlinearity_type=='linear':
            self.phi_to_yhat_layer = \
                AffineLayer(eta=self.eta_regularization,
                            monotonic=self.ge_nonlinearity_monotonic)
        elif self.ge_nonlinearity_type=='nonlinear':
            self.phi_to_yhat_layer = \
                GlobalEpistasisLayer(K=ge_nonlinearity_hidden_nodes,
                                     eta=self.eta_regularization,
                                     monotonic=self.ge_nonlinearity_monotonic)
        else:
            assert False, 'This shouldnt happen'

        yhat = self.phi_to_yhat_layer(phi)

        # Concatenate yhat and training labels
        yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
            [yhat, labels_input])

        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':
            self.noise_model_layer = GaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'Cauchy':
            self.noise_model_layer = CauchyNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'SkewedT':
            self.noise_model_layer = SkewedTNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)
        else:
            assert False, 'This should not happen.'

        outputTensor = self.noise_model_layer(yhat_y_concat)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model


@handle_errors
class MultiyGlobalEpistasisModel:
    """
    Represents a global epistatsis model for fitting
    multiple targets y (e.g. replicate measurements) at the
    same time.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox']

    ge_nonlinearity_type: (str)
        Specifies the form of the GE nonlinearity. Options:
        "linear": An affine transformation from phi to yhat.
        "nonlinear": Allow and arbitrary nonlinear map from phi to yhat.

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    number_of_replicate_targets: (int)
        Number of simultaneous targets to fit using 'Multi_y_GE' regression.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 alphabet,
                 gpmap_type,
                 gpmap_kwargs,
                 ge_nonlinearity_monotonic,
                 ge_nonlinearity_type,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization,
                 number_of_replicate_targets):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ge_nonlinearity_type = ge_nonlinearity_type
        self.number_of_replicate_targets = number_of_replicate_targets

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that ge_nonlinearity_type valid
        allowed_types = ['linear', 'nonlinear']
        check(self.ge_nonlinearity_type in allowed_types,
              f'gpmap_type = {self.gpmap_type};'
              f'must be in {allowed_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):
        """
        Establish model architecture.

        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        Parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        Returns
        -------
        model: (tf.model)
            TensorFlow model that can be compiled and subsequently fit to data.
        """
        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT'},
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

        number_input_layer_nodes = number_x_nodes + self.number_of_replicate_targets

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + self.number_of_replicate_targets],
            output_shape=((self.number_of_replicate_targets,)),
            trainable=False, name='Labels_input')(inputTensor)

        # list that contains information about replicate layers.
        replicates_input = []

        # build up lambda layers, on step at a time, which will be
        # fed to each of the measurement layers
        for replicate_layer_index in range(self.number_of_replicate_targets):

            print(replicate_layer_index, replicate_layer_index + 1)

            temp_replicate_layer = Lambda(lambda x:
                                          x[:, number_x_nodes + replicate_layer_index:
                                          number_x_nodes + replicate_layer_index + 1],
                                          output_shape=((1,)), trainable=False,
                                          name='Labels_input_' + str(replicate_layer_index))(inputTensor)

            replicates_input.append(temp_replicate_layer)

        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = AdditiveGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = PairwiseGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        elif self.gpmap_type == 'blackbox':
            self.x_to_phi_layer = MultilayerPerceptronGPMap(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                **self.gpmap_kwargs)
        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)


        # Make multi_y global epistasis layer
        if self.ge_nonlinearity_type == 'nonlinear':

            # phi feeds into each of the replicate intermediate layers
            self.phi_to_yhat_layer = []
            for intermediate_index in range(self.number_of_replicate_targets):

                # Make a GE layer for each target.
                temp_intermediate_layer = GlobalEpistasisLayer(K=ge_nonlinearity_hidden_nodes,
                                                               eta=self.eta_regularization,
                                                               monotonic=self.ge_nonlinearity_monotonic)

                self.phi_to_yhat_layer.append(temp_intermediate_layer(phi))

        elif self.ge_nonlinearity_type == 'linear':

            # phi feeds into each of the replicate intermediate layers
            self.phi_to_yhat_layer = []
            for intermediate_index in range(self.number_of_replicate_targets):
                # Make a GE layer for each target.
                temp_intermediate_layer = AffineLayer(eta=self.eta_regularization,
                                                      monotonic=self.ge_nonlinearity_monotonic)

                self.phi_to_yhat_layer.append(temp_intermediate_layer(phi))
        else:
            assert False, 'This shouldnt happen'

        # concatenate yhat_ith with labels_input_rep_ith into the likelihood layers
        # to compute the loss

        #yhat = self.phi_to_yhat_layer(phi)
        concatenateLayer_rep_input = []

        for concat_index in range(self.number_of_replicate_targets):
            temp_concat = Concatenate(name='yhat_and_y_to_ll' + str(concat_index)) \
                ([self.phi_to_yhat_layer[concat_index], replicates_input[concat_index]])

            concatenateLayer_rep_input.append(temp_concat)

        # Concatenate yhat and training labels
        # yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
        #     [yhat, labels_input])

        ll_rep_layers = []
        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':

            for ll_index in range(self.number_of_replicate_targets):

                print(f' Making Gaussian noise model, index = {ll_index}')
                temp_ll_layer = GaussianNoiseModelLayer(info_for_layers_dict=self.info_for_layers_dict,
                                                        polynomial_order=self.ge_heteroskedasticity_order,
                                                        eta_regularization=self.eta_regularization)

                ll_rep_layers.append(temp_ll_layer(concatenateLayer_rep_input[ll_index]))

        elif ge_noise_model_type == 'Cauchy':

            for ll_index in range(self.number_of_replicate_targets):

                temp_ll_layer = CauchyNoiseModelLayer(info_for_layers_dict=self.info_for_layers_dict,
                                                      polynomial_order=self.ge_heteroskedasticity_order,
                                                      eta_regularization=self.eta_regularization)

                ll_rep_layers.append(temp_ll_layer(concatenateLayer_rep_input[ll_index]))

        elif ge_noise_model_type == 'SkewedT':

            for ll_index in range(self.number_of_replicate_targets):

                temp_ll_layer = SkewedTNoiseModelLayer(info_for_layers_dict=self.info_for_layers_dict,
                                                       polynomial_order=self.ge_heteroskedasticity_order,
                                                       eta_regularization=self.eta_regularization)

                ll_rep_layers.append(temp_ll_layer(concatenateLayer_rep_input[ll_index]))


            # self.noise_model_layer = SkewedTNoiseModelLayer(
            #     info_for_layers_dict=self.info_for_layers_dict,
            #     polynomial_order=self.ge_heteroskedasticity_order,
            #     eta_regularization=self.eta_regularization)
        else:
            assert False, 'This should not happen.'

        #outputTensor = self.noise_model_layer(yhat_y_concat)
        self.noise_model_layer = ll_rep_layers
        outputTensor = self.noise_model_layer

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model



@handle_errors
class MultiPhiGlobalEpistasisModel:
    """
    Represents a global epistatsis model with multiple latent nodes and a single y (output).

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    number_latent_nodes: (int)
        Integer specifying the number of nodes in the first hidden layer.

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox']

    ge_nonlinearity_type: (str)
        Specifies the form of the GE nonlinearity. Options:
        "linear": An affine transformation from phi to yhat.
        "nonlinear": Allow and arbitrary nonlinear map from phi to yhat.

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 alphabet,
                 number_latent_nodes,
                 gpmap_type,
                 gpmap_kwargs,
                 ge_nonlinearity_monotonic,
                 ge_nonlinearity_type,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.number_latent_nodes = number_latent_nodes
        self.C = len(self.alphabet)
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ge_nonlinearity_type = ge_nonlinearity_type

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that ge_nonlinearity_type valid
        allowed_types = ['linear', 'nonlinear']
        check(self.ge_nonlinearity_type in allowed_types,
              f'gpmap_type = {self.gpmap_type};'
              f'must be in {allowed_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # check that number_latent_nodes is an number
        check(isinstance(self.number_latent_nodes, numbers.Integral),
              'number_latent_nodes must be an integer')

        # record sequence length for convenience
        self.L = sequence_length

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):
        """
        Establish model architecture.

        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        Parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        Returns
        -------
        model: (tf.model)
            TensorFlow model that can be compiled and subsequently fit to data.
        """
        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT'},
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

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

        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = Multi_AdditiveGPMapLayer(
                number_latent_nodes=self.number_latent_nodes,
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = Multi_PairwiseGPMapLayer(
                number_latent_nodes=self.number_latent_nodes,
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        # elif self.gpmap_type == 'blackbox':
        #     self.x_to_phi_layer = MultilayerPerceptronGPMap(
        #         L=self.L,
        #         C=self.C,
        #         theta_regularization=self.theta_regularization,
        #         **self.gpmap_kwargs)
        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Make global epistasis layer
        if self.ge_nonlinearity_type=='linear':
            self.phi_to_yhat_layer = \
                AffineLayer(eta=self.eta_regularization,
                            monotonic=self.ge_nonlinearity_monotonic)
        elif self.ge_nonlinearity_type=='nonlinear':
            self.phi_to_yhat_layer = \
                MultiPhiGlobalEpistasisLayer(K=ge_nonlinearity_hidden_nodes,
                                     eta=self.eta_regularization,
                                     monotonic=self.ge_nonlinearity_monotonic,
                                     number_latent_nodes=self.number_latent_nodes
                                     )
        else:
            assert False, 'This shouldnt happen'

        yhat = self.phi_to_yhat_layer(phi)

        # Concatenate yhat and training labels
        yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
            [yhat, labels_input])

        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':
            self.noise_model_layer = GaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'Cauchy':
            self.noise_model_layer = CauchyNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'SkewedT':
            self.noise_model_layer = SkewedTNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)
        else:
            assert False, 'This should not happen.'

        outputTensor = self.noise_model_layer(yhat_y_concat)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model


@handle_errors
class MeasurementProcessAgnosticModel:
    """
    Represents a measurement process agnostic model.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    number_of_bins: (int)
        Integer specifying the number of bins. (Only used for MPA regression).

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox', 'custom']

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    custom_gpmap: (GPMapLayer sub-class)
        Defines custom gpmap, provided by user. Inherited class of GP-MAP layer,
        which defines the functionality for x_to_phi_layer.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 number_of_bins,
                 gpmap_type,
                 gpmap_kwargs,
                 alphabet,
                 theta_regularization,
                 eta_regularization,
                 ohe_batch_size,
                 custom_gpmap):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.number_of_bins = number_of_bins
        self.custom_gpmap = custom_gpmap

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        self.mpa_hidden_nodes = None
        self.theta_gf = None
        self.na_model = None

        # check that L is an number
        check(isinstance(sequence_length, numbers.Integral),
              'L must be an integer')

        # check that number_of_bins is an number
        check(isinstance(self.number_of_bins, numbers.Integral),
              'number_of_bins must be an integer')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox', 'thermodynamic', 'custom']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

        # Record number of bins
        self.Y = number_of_bins
        self.all_y = np.arange(self.Y).astype(int)

    def define_model(self,
                     mpa_hidden_nodes=10):
        """
        Define the neural network architecture of the MPA model.

        Uses the tensorflow.keras functional API. If custom_architecture is
        not None, this is used instead as the model architecture.

        Parameters
        ----------
        mpa_hidden_nodes: (int)
            Number of nodes to use in the hidden layer of the measurement
            network of the GE model architecture.

        Returns
        -------
        model: (tf.model)
            A tensorflow model that can be compiled and subsequently
            fit to data.
        """
        check(isinstance(mpa_hidden_nodes, numbers.Integral),
              'mpa_hidden_nodes must be a number.')

        check(mpa_hidden_nodes > 0,
              'mpa_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)
        number_input_layer_nodes = number_x_nodes+self.number_of_bins

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + self.number_of_bins],
            output_shape=((1,)),
            trainable=False,
            name='Labels_input')(inputTensor)

        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = AdditiveGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = PairwiseGPMapLayer(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        elif self.gpmap_type == 'blackbox':
            self.x_to_phi_layer = MultilayerPerceptronGPMap(
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                **self.gpmap_kwargs)
        elif self.gpmap_type == 'thermodynamic':

            self.x_to_phi_layer = ThermodynamicGPMapLayer(
                C=self.C,
                **self.gpmap_kwargs)
        elif self.gpmap_type == 'custom':
            self.x_to_phi_layer = self.custom_gpmap(**self.gpmap_kwargs)

        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Create concatenation layer
        self.layer_concatenate_phi_ct = Concatenate(name='phi_and_ct')
        phi_ct = self.layer_concatenate_phi_ct([phi, labels_input])

        # Create measurement process layer
        self.layer_measurement_process = MPAMeasurementProcessLayer(
            info_for_layers_dict=self.info_for_layers_dict,
            Y=self.number_of_bins,
            K=mpa_hidden_nodes,
            eta=self.eta_regularization
            )
        outputTensor = self.layer_measurement_process(phi_ct)

        #create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.mpa_hidden_nodes = mpa_hidden_nodes
        return model


@handle_errors
class MultiMeasurementProcessAgnosticModel:
    """
    Represents a measurement process agnostic model with multiple
    latent phenotype nodes. Currently supports only additive latent
    trait models.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    number_of_bins: (int)
        Integer specifying the number of bins. (Only used for MPA regression).

    number_latent_nodes: (int)
        Integer specifying the number of nodes in the first hidden layer.

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive']

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 number_of_bins,
                 number_latent_nodes,
                 gpmap_type,
                 gpmap_kwargs,
                 alphabet,
                 theta_regularization,
                 eta_regularization,
                 ohe_batch_size):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.number_of_bins = number_of_bins
        self.number_latent_nodes = number_latent_nodes

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        self.mpa_hidden_nodes = None
        self.theta_gf = None
        self.na_model = None

        # check that L is an number
        check(isinstance(sequence_length, numbers.Integral),
              'L must be an integer')

        # check that L is an number
        check(isinstance(number_latent_nodes, numbers.Integral),
              'number_latent_nodes must be an integer')

        # check that number_of_bins is an number
        check(isinstance(self.number_of_bins, numbers.Integral),
              'number_of_bins must be an integer')

        # check that theta number_of_bins is greater than or equal to  1
        check(self.number_of_bins >= 1,
              'number_of_bins must be >= 1')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

        # Record number of bins
        self.Y = number_of_bins
        self.all_y = np.arange(self.Y).astype(int)

    def define_model(self,
                     mpa_hidden_nodes=10):
        """
        Define the neural network architecture of the MPA model.

        Uses the tensorflow.keras functional API. If custom_architecture is
        not None, this is used instead as the model architecture.

        Parameters
        ----------
        mpa_hidden_nodes: (int)
            Number of nodes to use in the hidden layer of the measurement
            network of the GE model architecture.

        Returns
        -------
        model: (tf.model)
            A tensorflow model that can be compiled and subsequently
            fit to data.
        """
        check(isinstance(mpa_hidden_nodes, numbers.Integral),
              'mpa_hidden_nodes must be a number.')

        check(mpa_hidden_nodes > 0,
              'mpa_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)
        number_input_layer_nodes = number_x_nodes+self.number_of_bins

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)

        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + self.number_of_bins],
            output_shape=((1,)),
            trainable=False,
            name='Labels_input')(inputTensor)

        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = Multi_AdditiveGPMapLayer(
                number_latent_nodes=self.number_latent_nodes,
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization)

        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = Multi_PairwiseGPMapLayer(
                number_latent_nodes=self.number_latent_nodes,
                L=self.L,
                C=self.C,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        # TODO: implement the following latent trait models
        # elif self.gpmap_type == 'blackbox':
        #     self.x_to_phi_layer = MultilayerPerceptronGPMap(
        #         L=self.L,
        #         C=self.C,
        #         theta_regularization=self.theta_regularization,
        #         **self.gpmap_kwargs)
        # else:
        #     assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Create concatenation layer
        self.layer_concatenate_phi_ct = Concatenate(name='phi_and_ct')
        phi_ct = self.layer_concatenate_phi_ct([phi, labels_input])

        # Create measurement process layer
        self.layer_measurement_process = MultiMPAMeasurementProcessLayer(
            info_for_layers_dict=self.info_for_layers_dict,
            Y=self.number_of_bins,
            K=mpa_hidden_nodes,
            eta=self.eta_regularization,
            L=self.number_latent_nodes
            )
        outputTensor = self.layer_measurement_process(phi_ct)

        #create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.mpa_hidden_nodes = mpa_hidden_nodes
        return model


@handle_errors
class AIEModel:
    """
    Represents an aggregation of individual effects model.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox']

    ge_nonlinearity_type: (str)
        Specifies the form of the GE nonlinearity. Options:
        "linear": An affine transformation from phi to yhat.
        "nonlinear": Allow and arbitrary nonlinear map from phi to yhat.

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 alphabet,
                 gpmap_type,
                 gpmap_kwargs,
                 ge_nonlinearity_monotonic,
                 ge_nonlinearity_type,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization,
                 custom_gpmap):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ge_nonlinearity_type = ge_nonlinearity_type
        self.custom_gpmap = custom_gpmap

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that model type valid
        # valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox']
        # check(self.gpmap_type in valid_gpmap_types,
        #       f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that ge_nonlinearity_type valid
        allowed_types = ['linear', 'nonlinear']
        check(self.ge_nonlinearity_type in allowed_types,
              f'gpmap_type = {self.gpmap_type};'
              f'must be in {allowed_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):
        """
        Establish model architecture.

        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        Parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

        Returns
        -------
        model: (tf.model)
            TensorFlow model that can be compiled and subsequently fit to data.
        """
        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT'},
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

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

        # create x to y-hat layer.
        self.x_to_yhat = self.custom_gpmap(**self.gpmap_kwargs)
        concatenateLayer = self.x_to_yhat(sequence_input)

        nonLinearLayer = Dense(ge_nonlinearity_hidden_nodes, activation='tanh')(concatenateLayer)
        nonLinearLayer = Dense(ge_nonlinearity_hidden_nodes, activation='tanh')(nonLinearLayer)
        yhat = Dense(1, activation='linear', name='yhat')(nonLinearLayer)

        # Concatenate yhat and training labels
        yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
            [yhat, labels_input])

        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':
            self.noise_model_layer = GaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'Cauchy':
            self.noise_model_layer = CauchyNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'SkewedT':
            self.noise_model_layer = SkewedTNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)
        else:
            assert False, 'This should not happen.'

        outputTensor = self.noise_model_layer(yhat_y_concat)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        return model