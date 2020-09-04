from mavenn.src.error_handling import handle_errors, check
from mavenn.src.UI import GlobalEpistasisModel, MeasurementProcessAgnosticModel
from mavenn.src.utils import fix_gauge_additive_model, fix_gauge_neighbor_model, fix_gauge_pairwise_model
from mavenn.src.utils import onehot_encode_array, \
    _generate_nbr_features_from_sequences, _generate_all_pair_features_from_sequences
from mavenn.src.likelihood_layers import *
from mavenn.src.utils import fixDiffeomorphicMode
from mavenn.src.utils import GaussianNoiseModel, CauchyNoiseModel, SkewedTNoiseModel
from mavenn.src.utils import mi_continuous, mi_mixed
from mavenn.src.utils import get_1pt_variants

# Needed for properly shaping outputs
from mavenn.src.utils import _shape_for_output, _get_shape_and_return_1d_array, _broadcast_arrays

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model as kerasFunctionalModel # to distinguish from class name
from tensorflow.keras.layers import Dense, Activation, Input, Lambda, Concatenate
from tensorflow.keras.constraints import non_neg as nonneg
import tensorflow.keras.backend as K

import pandas as pd
import numpy as np
import re


@handle_errors
class Model:

    """
    Mavenn's model class that lets the user choose either
    global epistasis regression or noise agnostic regression

    If regerssion_type == 'MPA', than ge_* parameters are not used.


    attributes
    ----------

    x: (array-like)
        Input pandas DataFrame containing sequences. x are
        DNA, RNA, or protein sequences to be regressed over

    y: (array-like)
        y represents counts in bins, or continuous measurement values
        corresponding to the sequences x

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein'].

    regression_type: (str)
        variable that choose type of regression, valid options
        include 'GE', 'MPA'

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Three possible choices allowed: ['additive','neighbor','pairwise']

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_nonlinearity_hidden_nodes:
        Number of hidden nodes (i.e. sigmoidal contributions) to use in the
        definition of the GE nonlinearity.

    ge_noise_model_type: (str)
        Specifies the type of noise model the user wants to infer.
        The possible choices allowed: ['Gaussian','Cauchy','SkewedT']

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model parameters
        dependent on y_hat, and thus render the noise model heteroskedastic. Set
        to zero for a homoskedastic noise model. (Only used for GE regression).

    na_hidden_nodes:
        Number of hidden nodes (i.e. sigmoidal contributions) to use in the
        definition of the MPA measurement process.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters $\theta$.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters $\eta$.

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    ct_n: (array-like of ints)
        For MPA regression only. List N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    """

    def __init__(self,
                 x,
                 y,
                 alphabet,
                 regression_type,
                 gpmap_type='additive',
                 ge_nonlinearity_monotonic=True,
                 ge_nonlinearity_hidden_nodes=50,
                 ge_noise_model_type='Gaussian',
                 ge_heteroskedasticity_order=2,
                 na_hidden_nodes=50,
                 theta_regularization=0.01,
                 eta_regularization=0.01,
                 ohe_batch_size=50000,
                 ct_n=None):

        # set class attributes
        self.x, self.y = x, y
        self.alphabet = alphabet
        self.regression_type = regression_type
        self.gpmap_type = gpmap_type
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes
        self.ge_noise_model_type = ge_noise_model_type
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.na_hidden_nodes = na_hidden_nodes
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.ct_n = ct_n

        # represents GE or MPA model object, depending which is chosen.
        # attribute value is set below
        self.model = None

        # check that regression_type is valid
        check(self.regression_type in {'MPA', 'GE'},
              'regression_type = %s; must be "MPA", or  "GE"' %
              self.gpmap_type)

        # choose model based on regression_type
        if regression_type == 'GE':

            self.model = GlobalEpistasisModel(X=self.x,
                                              y=self.y,
                                              gpmap_type=self.gpmap_type,
                                              ge_nonlinearity_monotonic=self.ge_nonlinearity_monotonic,
                                              alphabet=self.alphabet,
                                              ohe_batch_size=self.ohe_batch_size,
                                              ge_heteroskedasticity_order=self.ge_heteroskedasticity_order,
                                              theta_regularization=self.theta_regularization,
                                              eta_regularization=self.eta_regularization)

            self.define_model = self.model.define_model(ge_noise_model_type=self.ge_noise_model_type,
                                                        ge_nonlinearity_hidden_nodes=
                                                        self.ge_nonlinearity_hidden_nodes)

        elif regression_type == 'MPA':

            self.model = MeasurementProcessAgnosticModel(x=self.x,
                                                         y=self.y,
                                                         ct_n = self.ct_n,
                                                         alphabet=self.alphabet,
                                                         gpmap_type=self.gpmap_type,
                                                         theta_regularization=self.theta_regularization,
                                                         ohe_batch_size=self.ohe_batch_size)

            self.define_model = self.model.define_model(na_hidden_nodes=self.na_hidden_nodes)

    @handle_errors
    def gauge_fix_model_multiple_replicates(self):

        """
        Method that gauge fixes the model (x_to_phi+measurement).


        parameters
        ----------
        None

        returns
        -------
        None

        """

        # TODO disable this method if user uses custom architecture

        # Helper variables used for gauge fixing x_to_phi trait parameters theta below.
        sequence_length = len(self.model.x_train[0])
        alphabetSize = len(self.model.characters)

        # Non-gauge fixed theta
        theta_all = self.model.model.layers[2].get_weights()[0]    # E.g., could be theta_additive + theta_pairwise
        theta_nought = self.model.model.layers[2].get_weights()[1]
        theta = np.hstack((theta_nought, theta_all.ravel()))

        # The following conditionals gauge fix the x_to_phi parameters depending of the value of x_to_phi
        if self.gpmap_type == 'additive':

            # compute gauge-fixed, additive model theta
            theta_gf = fix_gauge_additive_model(sequence_length, alphabetSize, theta)

        elif self.gpmap_type == 'neighbor':

            # compute gauge-fixed, neighbor model theta
            theta_gf = fix_gauge_neighbor_model(sequence_length, alphabetSize, theta)

        elif self.gpmap_type == 'pairwise':

            # compute gauge-fixed, pairwise model theta
            theta_gf = fix_gauge_pairwise_model(sequence_length, alphabetSize, theta)

        # The following variable unfixed_gpmap is a tf.keras backend function
        # which computes the non-gauge fixed value of the hidden node phi for a given input
        # this is  used to compute diffeomorphic scaling factor.
        unfixed_gpmap = K.function([self.model.model.layers[1].input], [self.model.model.layers[2].output])

        # compute unfixed phi using the function unfixed_gpmap with training sequences.
        unfixed_phi = unfixed_gpmap([self.model.input_seqs_ohe])

        # Compute diffeomorphic scaling factor which is used to rescale the parameters theta
        diffeomorphic_std = np.sqrt(np.var(unfixed_phi[0]))
        diffeomorphic_mean = np.mean(unfixed_phi[0])

        # diffeomorphic_mode fix thetas
        theta_nought_gf = theta_gf[0]-diffeomorphic_mean
        thet_gf_vec = theta_gf[1:]/diffeomorphic_std

        # Default neural network weights that are non gauge fixed.
        # This will be used for updating the weights of the measurement
        # network after the gauge fixed neural network is define below.
        temp_weights = [layer.get_weights() for layer in self.model.model.layers]

        # define gauge fixed model
        if self.regression_type == 'GE':

            if len(self.model.y_train.shape) == 1:
                number_of_replicate_targets = 1
            else:
                number_of_replicate_targets = min(self.model.y_train.shape)

            print('number of y nodes to add: ', number_of_replicate_targets)
            # create input layer with nodes allowing sequence to be input and also
            # target labels to be input, together.
            #number_input_layer_nodes = len(self.input_seqs_ohe[0])+self.y_train.shape[0]
            number_input_layer_nodes = len(self.model.input_seqs_ohe[0]) + number_of_replicate_targets
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.model.input_seqs_ohe[0])],
                                    output_shape=((len(self.model.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)

            replicates_input = []

            #number_of_replicate_targets = self.y_train.shape[0]

            for replicate_layer_index in range(number_of_replicate_targets):

                # build up lambda layers, on step at a time, which will be
                # fed to each of the measurement blocks
                print(replicate_layer_index, replicate_layer_index + 1)

                temp_replicate_layer = Lambda(lambda x:
                                              x[:, len(self.model.input_seqs_ohe[0])+replicate_layer_index:
                                              len(self.model.input_seqs_ohe[0]) + replicate_layer_index + 1],
                                              output_shape=((1,)), trainable=False,
                                              name='Labels_input_'+str(replicate_layer_index))(inputTensor)

                replicates_input.append(temp_replicate_layer)

            # labels_input_rep1 = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0]):len(self.input_seqs_ohe[0]) + 1],
            #                       output_shape=((1, )), trainable=False, name='Labels_input_1')(inputTensor)
            #
            # labels_input_rep2 = Lambda(lambda x: x[:, len(self.input_seqs_ohe[0])+1:len(self.input_seqs_ohe[0]) + 2],
            #                            output_shape=((1,)), trainable=False, name='Labels_input_2')(inputTensor)

            # sequence to latent phenotype
            #phi = Dense(1, name='phi')(sequence_input)

        elif self.regression_type == 'MPA':

            number_input_layer_nodes = len(self.model.input_seqs_ohe[0])+self.model.y_train.shape[1]
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.model.input_seqs_ohe[0])],
                                    output_shape=((len(self.model.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
            labels_input = Lambda(lambda x: x[:, len(self.model.input_seqs_ohe[0]):len(self.model.input_seqs_ohe[0]) + self.model.y_train.shape[1]],
                                  output_shape=((1,)), trainable=False, name='Labels_input')(inputTensor)

        # same phi as before
        phi = Dense(1, name='phiPrime')(sequence_input)
        # fix diffeomorphic scale
        #phi_scaled = fixDiffeomorphicMode()(phi)
        phiOld = Dense(1, name='phi')(phi)

        # implement monotonicity constraints if GE regression
        if self.regression_type == 'GE':

            if self.ge_nonlinearity_monotonic==True:

                # phi feeds into each of the replicate intermediate layers
                intermediate_layers = []
                for intermediate_index in range(number_of_replicate_targets):

                    temp_intermediate_layer = Dense(self.ge_nonlinearity_hidden_nodes,
                                                    activation='sigmoid',
                                                    kernel_constraint=nonneg(),
                                                    name='intermediate_bbox_'+str(intermediate_index))(phiOld)

                    intermediate_layers.append(temp_intermediate_layer)

                yhat_layers = []
                for yhat_index in range(number_of_replicate_targets):

                    temp_yhat_layer = Dense(1, kernel_constraint=nonneg(),
                                            name='y_hat_rep_'+str(yhat_index))(intermediate_layers[yhat_index])
                    yhat_layers.append(temp_yhat_layer)

                # intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid',
                #                            kernel_constraint=nonneg())(phiOld)

                # y_hat = Dense(1, kernel_constraint=nonneg())(intermediateTensor)

                # concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([y_hat, labels_input])

                concatenateLayer_rep_input = []

                for concat_index in range(number_of_replicate_targets):

                    temp_concat = Concatenate(name='yhat_and_rep_'+str(concat_index))\
                        ([yhat_layers[concat_index], replicates_input[concat_index]])

                    concatenateLayer_rep_input.append(temp_concat)

                likelihoodClass = globals()[self.ge_noise_model_type + 'LikelihoodLayer']

                #ll_rep1 = likelihoodClass(self.polynomial_order_ll)(concatenateLayer_rep1)
                #ll_rep2 = likelihoodClass(self.polynomial_order_ll)(concatenateLayer_rep2)

                ll_rep_layers = []
                for ll_index in range(number_of_replicate_targets):
                    temp_ll_layer = likelihoodClass(self.ge_heteroskedasticity_order)(concatenateLayer_rep_input[ll_index])
                    ll_rep_layers.append(temp_ll_layer)


                #outputTensor = [ll_rep1, ll_rep2]
                outputTensor = ll_rep_layers

                # dynamic likelihood class instantiation by the globals dictionary
                # manual instantiation can be done as follows:
                # outputTensor = GaussianLikelihoodLayer()(concatenateLayer)

                # likelihoodClass = globals()[self.p_of_all_y_given_phi + 'LikelihoodLayer']
                # outputTensor = likelihoodClass(self.polynomial_order_ll)(concatenateLayer)

            else:
                intermediateTensor = Dense(self.ge_nonlinearity_hidden_nodes, activation='sigmoid')(phiOld)
                y_hat = Dense(1)(intermediateTensor)

                concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([y_hat, labels_input])

                likelihoodClass = globals()[self.ge_noise_model_type + 'LikelihoodLayer']
                outputTensor = likelihoodClass(self.ge_heteroskedasticity_order)(concatenateLayer)

        elif self.regression_type == 'MPA':

            #intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
            #outputTensor = Dense(np.shape(self.model.y_train[0])[0], activation='softmax')(intermediateTensor)

            intermediateTensor = Dense(self.ge_nonlinearity_hidden_nodes, activation='sigmoid')(phiOld)
            yhat = Dense(np.shape(self.model.y_train[0])[0], name='yhat', activation='softmax')(intermediateTensor)

            concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
            outputTensor = MPALikelihoodLayer(number_bins=np.shape(self.model.y_train[0])[0])(concatenateLayer)


        # create the gauge-fixed model:
        model_gf = kerasFunctionalModel(inputTensor, outputTensor)

        # set new model theta weights
        theta_nought_gf = theta_nought_gf
        model_gf.layers[2].set_weights([thet_gf_vec.reshape(-1, 1), np.array([theta_nought_gf])])

        # update weights as sigma*phi+mean, which ensures predictions (y_hat) don't change from
        # the diffeomorphic scaling.
        model_gf.layers[3].set_weights([np.array([[diffeomorphic_std]]), np.array([diffeomorphic_mean])])

        for layer_index in range(4, len(model_gf.layers)):
            model_gf.layers[layer_index].set_weights(temp_weights[layer_index-1])

        # Update default neural network model with gauge-fixed model
        self.model.model = model_gf

        # The theta_gf attribute now contains gauge fixed parameters, and
        # can be obtained in raw form by accessing this attribute or can be
        # obtained a readable format by using the method return_theta
        self.model.theta_gf = theta_gf.reshape(len(theta_gf), 1)

    @handle_errors
    def fix_gauge(self, gauge="hierarchichal", wt_sequence=None):
        """
        Gauge-fixes the G-P map parameters $\theta$

        parameters
        ----------
        gauge: (string)
            Gauge to use. Options are "hierarchichal" or "wild-type".

        wt_sequence: (string)
            Sequence to use when adopting the wild-type gauge.

        returns
        -------
        None
        """
        # TODO: Fill out this function. If user calls this method and wants
        # to switch to WT gauge, if parameters are HA, switch parameters to WT, and vice versa
        self.gauge = gauge
        self.wt_sequence = wt_sequence
        pass

    # TODO: put underscore in front on function name
    @handle_errors
    def gauge_fix_model(self,
                        load_model=False,
                        diffeomorphic_mean=None,
                        diffeomorphic_std=None):

        """
        Method that gauge fixes the entire model (x_to_phi+measurement).

        parameters
        ----------
        load_model: (bool)
            If true, then this variable specifies that this method was used while calling load(),
            else, this method is called during fit. The purpose of calling this model during load
            is to ensure that it has the appropriate model architecture. However this variable
            ensures that the theta parameters aren't rescaled again.

        returns
        -------
        None

        """

        # Helper variables used for gauge fixing x_to_phi trait parameters theta below.
        sequence_length = len(self.model.x_train[0])
        alphabetSize = len(self.model.characters)

        # Non-gauge fixed theta
        theta_all = self.model.model.layers[2].get_weights()[0]    # E.g., could be theta_additive + theta_pairwise
        theta_nought = self.model.model.layers[2].get_weights()[1]
        theta = np.hstack((theta_nought, theta_all.ravel()))

        # The following conditionals gauge fix the x_to_phi parameters depending of the value of x_to_phi
        if self.gpmap_type == 'additive':

            # compute gauge-fixed, additive model theta
            theta_gf = fix_gauge_additive_model(sequence_length, alphabetSize, theta)

        elif self.gpmap_type == 'neighbor':

            # compute gauge-fixed, neighbor model theta
            theta_gf = fix_gauge_neighbor_model(sequence_length, alphabetSize, theta)

        elif self.gpmap_type == 'pairwise':

            # compute gauge-fixed, pairwise model theta
            theta_gf = fix_gauge_pairwise_model(sequence_length, alphabetSize, theta)

        # The following variable unfixed_gpmap is a tf.keras backend function
        # which computes the non-gauge fixed value of the hidden node phi for a given input
        # this is  used to compute diffeomorphic scaling factor.
        unfixed_gpmap = K.function([self.model.model.layers[1].input], [self.model.model.layers[2].output])

        # compute unfixed phi using the function unfixed_gpmap with training sequences.
        unfixed_phi = unfixed_gpmap([self.model.input_seqs_ohe])

        # if load model is false, record the following attributes which will be used when loading model
        if load_model==False:

            # Compute diffeomorphic scaling factor which is used to rescale the parameters theta
            diffeomorphic_std = np.sqrt(np.var(unfixed_phi[0]))
            diffeomorphic_mean = np.mean(unfixed_phi[0])

            # ensure ymean is an increasing function of phi (change made on 09/04/2020)
            if self.regression_type == 'MPA':
                # compute for training phi

                # Note, can't use function self.na_p_of_all_y_given_phi since model isn't gauge fixed yet.
                na_model_input = Input((1,))
                next_input = na_model_input

                # the following variable is the index of
                phi_index = 3
                yhat_index = 5

                # Form model using functional API in a loop, starting from
                # phi input, and ending on network output
                for layer in self.model.model.layers[phi_index:yhat_index]:
                    next_input = layer(next_input)

                # Form gauge fixed GE_nonlinearity model
                temp_na_model = kerasFunctionalModel(inputs=na_model_input, outputs=next_input)

                # compute the value of the nonlinearity for a given phi

                p_of_all_y_given_phi = temp_na_model.predict([unfixed_phi[0]])
                bin_numbers = np.arange(p_of_all_y_given_phi.shape[1])

                ymean = p_of_all_y_given_phi @ bin_numbers
                r = np.corrcoef(unfixed_phi[0].ravel(), ymean)[0, 1]

            elif self.regression_type == 'GE':

                r = np.corrcoef(unfixed_phi[0].ravel(), self.model.y_train.ravel())[0, 1]

            # this ensures phi is positively correlated with y_mean or y_train (MPA, GE respectively).
            if r < 0:
                diffeomorphic_std = -diffeomorphic_std

            # these attributes will also be saved in the saved model config file.
            self.diffeomorphic_mean = diffeomorphic_mean
            self.diffeomorphic_std = diffeomorphic_std


        # if this method is called after fit, scale the parameters to fix diffeomorphic mode.
        if load_model==False:
            # diffeomorphic_mode fix thetas
            theta_nought_gf = theta_gf[0]-diffeomorphic_mean
            theta_nought_gf/=diffeomorphic_std
            thet_gf_vec = theta_gf[1:]/diffeomorphic_std
        # if model is called during model load, then parameters were already scaled.
        else:
            theta_nought_gf = theta_gf[0]
            thet_gf_vec = theta_gf[1:]

        # Default neural network weights that are non gauge fixed.
        # This will be used for updating the weights of the measurement
        # network after the gauge fixed neural network is define below.
        temp_weights = [layer.get_weights() for layer in self.model.model.layers]

        # define gauge fixed model

        if self.regression_type == 'GE':

            number_input_layer_nodes = len(self.model.input_seqs_ohe[0]) + 1     # the plus 1 indicates the node for y
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.model.input_seqs_ohe[0])],
                                    output_shape=((len(self.model.input_seqs_ohe[0]),)))(inputTensor)

            labels_input = Lambda(
                lambda x: x[:, len(self.model.input_seqs_ohe[0]):len(self.model.input_seqs_ohe[0]) + 1],
                output_shape=((1,)), trainable=False)(inputTensor)

        elif self.regression_type == 'MPA':

            number_input_layer_nodes = len(self.model.input_seqs_ohe[0])+self.model.y_train.shape[1]
            inputTensor = Input((number_input_layer_nodes,), name='Sequence_labels_input')

            sequence_input = Lambda(lambda x: x[:, 0:len(self.model.input_seqs_ohe[0])],
                                    output_shape=((len(self.model.input_seqs_ohe[0]),)), name='Sequence_only')(inputTensor)
            labels_input = Lambda(lambda x: x[:, len(self.model.input_seqs_ohe[0]):len(self.model.input_seqs_ohe[0]) + self.model.y_train.shape[1]],
                                  output_shape=((1,)), trainable=False, name='Labels_input')(inputTensor)



        # same phi as before
        phi = Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(self.theta_regularization),
                    name='phiPrime')(sequence_input)
        # fix diffeomorphic scale
        #phi_scaled = fixDiffeomorphicMode()(phi)
        phiOld = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.theta_regularization), name='phi')(phi)

        # implement monotonicity constraints if GE regression
        if self.regression_type == 'GE':

            if self.ge_nonlinearity_monotonic==True:

                intermediateTensor = Dense(self.ge_nonlinearity_hidden_nodes, activation='sigmoid',
                                           kernel_constraint=nonneg())(phiOld)
                y_hat = Dense(1, kernel_constraint=nonneg())(intermediateTensor)

                concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([y_hat, labels_input])

                # dynamic likelihood class instantiation by the globals dictionary
                # manual instantiation can be done as follows:
                # outputTensor = GaussianLikelihoodLayer()(concatenateLayer)

                likelihoodClass = globals()[self.ge_noise_model_type + 'LikelihoodLayer']
                outputTensor = likelihoodClass(self.ge_heteroskedasticity_order, self.eta_regularization)(concatenateLayer)

            else:
                intermediateTensor = Dense(self.ge_nonlinearity_hidden_nodes, activation='sigmoid')(phiOld)
                y_hat = Dense(1)(intermediateTensor)

                concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([y_hat, labels_input])

                likelihoodClass = globals()[self.ge_noise_model_type + 'LikelihoodLayer']
                outputTensor = likelihoodClass(self.ge_heteroskedasticity_order, self.eta_regularization)(concatenateLayer)

        elif self.regression_type == 'MPA':

            #intermediateTensor = Dense(self.num_nodes_hidden_measurement_layer, activation='sigmoid')(phi)
            #outputTensor = Dense(np.shape(self.model.y_train[0])[0], activation='softmax')(intermediateTensor)

            intermediateTensor = Dense(self.na_hidden_nodes, activation='sigmoid')(phiOld)
            yhat = Dense(np.shape(self.model.y_train[0])[0], name='yhat', activation='softmax')(intermediateTensor)

            concatenateLayer = Concatenate(name='yhat_and_y_to_ll')([yhat, labels_input])
            outputTensor = MPALikelihoodLayer(number_bins=np.shape(self.model.y_train[0])[0])(concatenateLayer)


        # create the gauge-fixed model:
        model_gf = kerasFunctionalModel(inputTensor, outputTensor)

        # set new model theta weights
        theta_nought_gf = theta_nought_gf
        model_gf.layers[2].set_weights([thet_gf_vec.reshape(-1, 1), np.array([theta_nought_gf])])

        # update weights as sigma*phi+mean, which ensures predictions (y_hat) don't change from
        # the diffeomorphic scaling.
        model_gf.layers[3].set_weights([np.array([[diffeomorphic_std]]), np.array([diffeomorphic_mean])])

        for layer_index in range(4, len(model_gf.layers)):
            model_gf.layers[layer_index].set_weights(temp_weights[layer_index-1])

        # Update default neural network model with gauge-fixed model
        self.model.model = model_gf

        # The theta_gf attribute now contains gauge fixed parameters, and
        # can be obtained in raw form by accessing this attribute or can be
        # obtained a readable format by using the method return_theta
        self.model.theta_gf = theta_gf.reshape(len(theta_gf), 1)


    @handle_errors
    def fit(self,
            epochs=50,
            learning_rate=0.005,
            validation_split=0.2,
            verbose=1,
            early_stopping=True,
            early_stopping_patience=20,
            callbacks=[],
            optimizer=Adam,
            optimizer_kwargs={},
            fit_kwargs={},
            compile_kwargs={}):

        """

        Infers parameters, from data, for both the G-P map and the measurement process.

        parameters
        ----------
        epochs: (int>0)
            Maximum number of epochs to complete during training.

        learning_rate: (float > 0)
            Learning rate that will get passed to the optimizer.

        validation_split: (float in [0,1])
            Fraction of training data to be split into a validation set.

        verbose: (0 or 1, or boolean)
            Will show training progress if 1 or True, nothing if 0 or False.

        early_stopping: (bool)
            specifies whether to use early stopping or not

        early_stopping_patience: (int)
            If using early stopping, specifies the number of epochs to wait
            after a new optimum is identified.


        callbacks: (list)
            List of tf.keras.callbacks.Callback instances.

        optimizer: (string or tf.keras.optimizers.Optimizer instance)
            Optimizer to use. Name of a TensorFlow optimizer. Valid string options are:
            ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

        optimizer_kwargs: (dict)
            Additional keyword arguments to pass to the constructor of the
            tf.keras.optimizers.Optimizer class.

        fit_kwargs: (dict)
            Additional keyword arguments to pass to tf.keras.model.fit().

        compile_kwargs: (dict):
            Additional keyword arguments to pass to tf.keras.model.compile().

        returns
        -------
        history: (tf.keras.callbacks.History object)
            Standard TensorFlow record of the optimization session.

        """

        self.learning_rate = learning_rate

        # removing compiler kwargs temporarily to debug RTD issues.
        self._compile_model(optimizer=optimizer,
                           lr=self.learning_rate,
                           **optimizer_kwargs)
                           #**compile_kwargs)

        if early_stopping:
            callbacks += [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  mode='auto',
                                                                  patience=early_stopping_patience)]

        # OHE training sequences with y appended to facilitate the calculation of likelihood.
        train_sequences = []

        # To each sequence in the training set, its target value is appended
        # to its one-hot encoded form, which gets passed to fit.
        for n in range(len(self.model.input_seqs_ohe)):
            temp = self.model.input_seqs_ohe[n].ravel()
            temp = np.append(temp, self.model.y_train[n])
            train_sequences.append(temp)

        train_sequences = np.array(train_sequences)

        history = self.model.model.fit(train_sequences,
                                       self.model.y_train,
                                       validation_split=validation_split,
                                       epochs=epochs,
                                       verbose=verbose,
                                       callbacks=callbacks,
                                       #**fit_kwargs,
                                       )

        # gauge fix model after fitting
        self.gauge_fix_model()

        # update history attribute
        self.model.history = history
        return history

    @handle_errors
    def phi_to_yhat(self,
                    phi):

        """
        Evaluate the GE nonlinearity at specified values of phi (the latent phenotype).

        parameters
        ----------

        phi: (array-like)
            Latent phenotype values at which to evaluate the GE nonlinearity

        returns
        -------
        y_hat: (array-like)
            GE nonlinearity evaluated on phi values

        """

        # Shape phi for processing
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        check(self.regression_type == 'GE', 'regression type must be "GE" for this function ')

        yhat = self.model.phi_to_yhat(phi)

        # Shape yhat for output
        yhat = _shape_for_output(yhat, phi_shape)

        return yhat

    @handle_errors
    def get_gpmap_parameters(self):

        """
        Returns gauge-fixed parameters for the G-P map as a pandas dataframe.
        The returned dataframe has two columns: name and values.
        The format of each parameter name is of the following form:

        constant parameters: "theta_0"
        additive parameters: "theta_1:A"
        pairwise parameters: "theta_1:A,5:G"

        returns
        -------
        theta_df: (pd.DataFrame)
            Gauge-fixed G-P map parameters, formatted as a dataframe.
        """

        # temp variable to store characters.
        chars = self.model.characters

        # position and character indices
        char_indices = list(range(len(chars)))
        pos_indices = list(range(len(self.model.x_train[0])))

        # update theta_gf in case load model is called.
        theta_0 = self.get_nn().layers[2].get_weights()[1]
        theta_gpmap = self.get_nn().layers[2].get_weights()[0]
        self.model.theta_gf = np.insert(theta_gpmap, 0, theta_0)


        # list that will contain parameter names
        names = []

        # list that will contain parameter values corresponding to names
        values = []

        # These parameters are gauge fixed are the model has been fit.
        if self.gpmap_type == 'additive':

            # get constant term.
            #print(self.model.theta_gf.shape)
            theta_0 = self.model.theta_gf[0]

            # add it to the lists that will get returned.
            names.append('theta_0')
            values.append(theta_0)

            reshaped_theta = self.model.theta_gf[1:].reshape(len(self.model.x_train[0]), len(chars))
            for position in pos_indices:
                for char in char_indices:
                    names.append('theta_' + str(position) + ':' + chars[char])
                    values.append(reshaped_theta[position][char])

        elif self.gpmap_type == 'neighbor':

            # define helper variables
            sequenceLength = len(self.model.x_train[0])
            num_possible_pairs = int((sequenceLength * (sequenceLength - 1)) / 2)

            # get constant term.
            theta_0 = self.model.theta_gf[0]

            # add it to the lists that will get returned.
            names.append('theta_0')
            values.append(theta_0)

            # get additive terms, starting from 1 because 0 represents constant term
            reshaped_theta = self.model.theta_gf[1:sequenceLength*len(self.model.characters)+1].\
                reshape(len(self.model.x_train[0]), len(chars))

            for position in pos_indices:
                for char in char_indices:
                    names.append('theta_' + str(position) + ':' + chars[char])
                    values.append(reshaped_theta[position][char])

            reshaped_theta = self.model.theta_gf[sequenceLength*len(self.model.characters)+1:]\
                .reshape(len(self.model.x_train[0]) - 1, len(chars), len(chars))

            # get parameters in tidy format
            for pos1 in pos_indices[:-1]:
                for char1 in char_indices:
                    for char2 in char_indices:
                        value = reshaped_theta[pos1][char1][char2]
                        name = f'theta_{pos1}:{chars[char1]}{chars[char2]}'
                        names.append(name)
                        values.append(value)

        elif self.gpmap_type == 'pairwise':

            # define helper variables
            sequenceLength = len(self.model.x_train[0])
            num_possible_pairs = int((sequenceLength * (sequenceLength - 1)) / 2)

            # get constant term.
            theta_0 = self.model.theta_gf[0]

            # add it to the lists that will get returned.
            names.append('theta_0')
            values.append(theta_0)

            # get additive terms, starting from 1 because 0 represents constant term
            reshaped_theta = self.model.theta_gf[1:sequenceLength*len(self.model.characters)+1].\
                reshape(len(self.model.x_train[0]), len(chars))

            for position in pos_indices:
                for char in char_indices:
                    names.append('theta_' + str(position) + ':' + chars[char])
                    values.append(reshaped_theta[position][char])


            # get pairwise terms
            # reshape to num_possible_pairs by len(chars) by len(chars) array
            reshaped_theta = self.model.theta_gf[sequenceLength*len(self.model.characters)+1:].\
                reshape(num_possible_pairs, len(chars), len(chars))

            pos_pair_num = 0
            for pos1 in pos_indices:
                for pos2 in pos_indices[pos1+1:]:
                    for char1 in char_indices:
                        for char2 in char_indices:
                            value = reshaped_theta[pos_pair_num][char1][char2]
                            name = f'theta_{pos1}:{chars[char1]},{pos2}:{chars[char2]}'
                            names.append(name)
                            values.append(value)
                    pos_pair_num += 1

        theta_df = pd.DataFrame(
            {'name': names,
             'value': values
             })

        return theta_df

    @handle_errors
    def get_nn(self):

        """
        Returns the tf neural network used to represent the inferred model.
        """

        return self.model.model

    # TODO: Make internal
    @handle_errors
    def _compile_model(self,
                      optimizer=Adam,
                      lr=0.005,
                      optimizer_kwargs={},
                      compile_kwargs={}):
        """
        This method will compile the model created in the constructor. The loss used will be
        log_poisson_loss for MPA regression, or mean_squared_error for GE regression

        parameters
        ----------

        optimizer: (str)
            Specifies which optimizers to use during training. See
            'https://www.tensorflow.org/api_docs/python/tf/keras/optimizers',
            for a all available optimizers


        lr: (float)
            Learning rate of the optimizer.

        returns
        -------
        None

        """

        if self.regression_type == 'GE':

            # Note: this loss just returns the computed
            # Likelihood in the custom likelihood layer
            def likelihood_loss(y_true, y_pred):

                return K.sum(y_pred)

            self.model.model.compile(loss=likelihood_loss,
                                     optimizer=optimizer(lr=lr, **optimizer_kwargs),
                                     **compile_kwargs)

        elif self.regression_type == 'MPA':


            def likelihood_loss(y_true, y_pred):
                return y_pred

            #self.model.model.compile(loss=tf.nn.log_poisson_loss,
            self.model.model.compile(loss=likelihood_loss,
                                     optimizer=optimizer(lr=lr, **optimizer_kwargs),
                                     **compile_kwargs)

    @handle_errors
    def x_to_phi(self,
                 x):
        """

        Evaluates the latent phenotype phi on input sequences.

        parameters
        ----------
        x: (array-like of str)
            Sequence inputs representing DNA, RNA, or protein (whichever
            type of sequence the model was trained on). Input can must be
            an array of str, all the proper length.

        returns
        -------
        phi: (array-like of float)
            Array of latent phenotype values.

        """

        # Shape x for processing
        x, x_shape = _get_shape_and_return_1d_array(x)

        # Make sure all sequences have the proper length
        L = self.model.L
        lengths = np.unique([len(seq) for seq in x])
        check(len(lengths) == 1,
              f'Input sequences have multiple lengths: {lengths}')
        check(lengths[0] == L,
              f'Input sequence length {lengths[0]} does not match L={L}')

        # Make sure sequences in x have only valid characters
        model_chars = set(self.model.characters)
        x_chars = set(''.join(x))
        check(x_chars <= model_chars,
              f'Input sequences contain the following invalid characters: {x_chars-model_chars}')

        if self.gpmap_type == 'additive':
            # one-hot encode sequences in batches in a vectorized way
            seqs_ohe = onehot_encode_array(x, self.model.characters)

        elif self.gpmap_type == 'neighbor':
            # Generate additive one-hot encoding.
            X_test_additive = onehot_encode_array(x, self.model.characters, self.ohe_batch_size)

            # Generate neighbor one-hot encoding.
            X_test_neighbor = _generate_nbr_features_from_sequences(x, self.alphabet)

            # Append additive and neighbor features together.
            seqs_ohe = np.hstack((X_test_additive, X_test_neighbor))

        elif self.gpmap_type == 'pairwise':
            # Generate additive one-hot encoding.
            X_test_additive = onehot_encode_array(x, self.model.characters, self.ohe_batch_size)

            # Generate pairwise one-hot encoding.
            X_test_pairwise = _generate_all_pair_features_from_sequences(x, self.alphabet)

            # Append additive and pairwise features together.
            seqs_ohe = np.hstack((X_test_additive, X_test_pairwise))

        # Form tf.keras function that will evaluate the value of gauge fixed latent phenotype
        gpmap_function = K.function([self.model.model.layers[1].input], [self.model.model.layers[2].output])

        # Compute latent phenotype values
        phi = gpmap_function([seqs_ohe])

        # Remove extra dimension tf adds
        #phi = phi[0].ravel().copy()

        # Shape phi for output
        phi = _shape_for_output(phi, x_shape)

        # Return latent phenotype values
        return phi

    @handle_errors
    def x_to_yhat(self,
                  x):

        """
        Make predictions for arbitrary input sequences. Note that this returns the output of
        the measurement process, not the latent phenotype.

        parameters
        ----------
        x: (array-like)
            Sequence data on which to make predictions.

        returns
        -------
        predictions: (array-like)
            An array of predictions for GE regression.
        """

        # Shape x for processing
        x, x_shape = _get_shape_and_return_1d_array(x)

        check(self.regression_type == 'GE', 'Regression type must be GE for this function.')

        yhat = self.phi_to_yhat(self.x_to_phi(x))

        #yhat = yhat[0].ravel().copy()

        # Shape yhat for output
        yhat = _shape_for_output(yhat, x_shape)

        return yhat


    def I_predictive(self,
                     x,
                     y,
                     knn=5,
                     uncertainty=True,
                     num_subsamples=25,
                     use_LNC=False,
                     alpha_LNC=.5,
                     verbose=False):
        """
        Estimate the predictive information I[y;phi] on supplied data.

        parameters
        ----------

        x: (array-like of strings)
            Array of sequences for which to comptue phi values.

        y: (array-like of floats)
            Array of measurements y to use when computing I[y;phi].
            If measurements are continuous, y must be the same shape as
            x. If measurements are discrete, y can have two formats.
            If y_format="list", y should be a list of discrete values,
            one for each x. If y_format="matrix", y should be a
            MxY matrix, where M=len(x) and Y is the number of possible
            values for Y.

        knn: (int>0)
            Number of nearest neighbors to use in the KSG estimator.

        uncertainty: (bool)
            Whether to estimate the uncertainty of the MI estimate.
            Substantially increases runtime if True.

        num_subsamples: (int > 0)
            Number of subsamples to use if estimating uncertainty.

        use_LNC: (bool)
            Whether to compute the Local Nonuniform Correction
            (LNC) using the method of Gao et al., 2015.
            Substantially increases runtime if True. Only used for
            continuous y values.

        alpha_LNC: (float in (0,1))
            Value of alpha to use when computing LNC.
            See Gao et al., 2015 for details. Only used for
            continuous y values.

        verbose: (bool)
            Whether to print results and execution time.

        returns
        -------

        (I, dI): (float, float)
            I = Mutual information estimate in bits.
            dI = Uncertainty estimate in bits. Zero if uncertainty=False is set.
            Not returned if uncertainty=False is set.

        """

        if self.regression_type=='GE':
            return mi_continuous(self.x_to_phi(x),
                                 y,
                                 knn=knn,
                                 uncertainty=uncertainty,
                                 use_LNC=use_LNC,
                                 alpha_LNC=alpha_LNC,
                                 verbose=verbose)

        elif self.regression_type=='MPA':

            phi = self.x_to_phi(x)

            # The format of y needs to be integer bin numbers, like the input to mavenn
            # for MPA regression

            return mi_mixed(phi,
                            y,
                            knn=knn,
                            uncertainty=uncertainty,
                            num_subsamples=num_subsamples,
                            verbose=verbose)

    def yhat_to_yq(self,
                   yhat,
                   q=[0.16,0.84]):
        """
        Returns quantile values of p(y|yhat) given yhat and the quantiles q.
        Reserved only for GE models

        parameters
        ----------

        yhat: (array of floats)
            Values from which p(y|yhat) is computed.

        q: (array of floats in [0,1])
            Quantile specifications

        returns
        -------

        yq: (array of floats)
            Array of quantile values.
        """

        # Shape yhat for processing
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)

        # Shape x for processing
        q, q_shape = _get_shape_and_return_1d_array(q)

        check(self.regression_type=='GE', 'regression type must be GE for this methdd')
        # Get GE noise model based on the users input.
        # 20.09.03 JBK: I don't understand this line.
        yqs = globals()[self.ge_noise_model_type + 'NoiseModel'](self,yhat,q=q).user_quantile_values

        # This seems to be needed
        yqs = np.array(yqs).T

        # Shape yqs for output
        yqs_shape = yhat_shape + q_shape
        yqs = _shape_for_output(yqs, yqs_shape)

        return yqs



    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Computes the p(y|phi) for both GE and MPA regression.

        y: (number or array-like of numbers)
            Measurement values. Note that these are cast as integers for
            MPA regression.

        phi: (float or array-like of floats)
            Latent phenotype values.

        paired: (bool)
            Whether y,phi values should be treated as pairs.
            If so, y and phi must have the same number of elements.
            The shape of y will be used as output.

        returns
        -------
            p: (float or array-like of floats)
                Probability of y given phi.
        """

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Do computation
            p = self._p_of_y_given_phi(y, phi)

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Do computation
            p = self._p_of_y_given_phi(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p


    # TODO: Stated behavior won't work for MPA regression, only GE
    def _p_of_y_given_phi(self,
                         y,
                         phi):

        """
        Method that computes the p(y|phi) for both GE and MPA regression.

        Note that if y is and np.ndarray with shape=(n1,n2,...,nK) and
        phi is an np.ndarray with shape=(n1,n2,...,nK), the returned value
        p_of_y_given_phi will also have shape=(n1,n2,...,nK). In other
        cases, the appropriate broadcasting will occur.

        y: (float (GE) or int (MPA))
            Specifies continuous target value for GE regression or an integer
            specifying bin number for MPA regression.

        phi: (float)
            Latent phenotype on which probability is conditioned.

        returns
        -------
        p_of_y_given_phi: (float)
            Probaility of y given phi.

        """

        # Reshape inputs for processing
        #y, y_shape = _get_shape_and_return_1d_array(y)
        #phi, phi_shape = _get_shape_and_return_1d_array(y)

        if self.regression_type == 'MPA':

            # # check that entered y (specifying bin number) is an integer
            # check(isinstance(y, int),
            #       'type(y), specifying bin number, must be of type int')
            #
            # # check that entered bin nnumber doesn't exceed max bins
            # check(y< self.model.y_train[0].shape[0],
            #       "bin number cannot be larger than max bins = %d" %self.model.y_train[0].shape[0])
            #
            # # Give the probability of bin y given phi, note phi can be an array.
            # p_of_y_given_phi = self.na_p_of_all_y_given_phi(phi)[:,y]
            # #return p_of_y_given_phi

            in_shape = y.shape

            # Cast y as integers
            y = y.astype(int)

            # Make sure all y values are valid
            Y = self.model.y_train[0].shape[0]
            check(np.all(y >= 0),
                  f"Negative values for y are invalid for MAP regression")

            check(np.all(y < Y),
                  f"Some y values exceed the number of bins {Y}")

            # Have to ravel
            y = y.ravel()
            phi = phi.ravel()

            # Get values for all bins
            p_of_all_y_given_phi = self.na_p_of_all_y_given_phi(phi)

            # There has to be a better way to do this
            p_of_y_given_phi = np.zeros(len(y))
            for i, _y in enumerate(y):
                p_of_y_given_phi[i] = p_of_all_y_given_phi[i, _y]

            # Reshape
            p_of_y_given_phi = np.reshape(p_of_y_given_phi, in_shape)


        else:
            check(self.regression_type=='GE',
                  f'Invalid regression type {self.regression_type}.')

            # variable to store the shape of the returned object

            yhat = self.phi_to_yhat(np.array(phi).ravel())

            if np.array(y).shape==np.array(phi).shape and (len(np.array(y).shape)>0 and len(np.array(phi).shape)>0):

                shape = np.array(y).shape

                p_of_y_given_phi = self._p_of_y_given_y_hat(y.ravel(), yhat.ravel()).reshape(shape)

                #return p_of_y_given_phi

            else:

                p_of_y_given_phi = self._p_of_y_given_y_hat(y, yhat)

                #return p_of_y_given_phi

        # Reshape for output
        #p_shape = y_shape + phi_shape
        #p = _shape_for_output(p_of_y_given_phi, p_shape)
        #return p
        return p_of_y_given_phi


    def p_of_y_given_yhat(self, y, yhat, paired=False):
        """
        Computes the p(y|yhat) for GE only.

        y: (float or array-like of floats)
            Measurement values.

        yhat: (float or array-like of floats)
            Latent phenotype values.

        paired: (bool)
            Whether y,yhat values should be treated as pairs.
            If so, y and yhat must have the same number of elements.
            The shape of y will be used as output.

        returns
        -------
            p: (float or array-like of floats)
                Probability of y given yhat.
        """

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        yhat, yhat_shape = _get_shape_and_return_1d_array(yhat)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == yhat_shape,
                  f"y shape={y_shape} does not match yhat shape={yhat_shape}")

            # Do computation
            p = self._p_of_y_given_y_hat(y, yhat)

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and yhat
            y, yhat = _broadcast_arrays(y, yhat)

            # Do computation
            p = self._p_of_y_given_y_hat(y, yhat)

            # Set output shape
            p_shape = y_shape + yhat_shape

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p



    def _p_of_y_given_y_hat(self,
                            y,
                            yhat):

        """
        Method that returns computes.

        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.

        yhat: (float)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p_of_y_given_yhat: (array-like of floats)
            Probability of y given sequence yhat. Shape of returned value will
            match shape of y_test, for a single yhat. For each value of yhat_i,
            the distribution p(y|yhat_i), where i traverses the elements of yhat.

        """

        check(self.regression_type=='GE', "This method works on with GE regression.")
        # Get GE noise model based on the users input.
        ge_noise_model = globals()[self.ge_noise_model_type + 'NoiseModel'](self, yhat, None)

        return ge_noise_model.p_of_y_given_yhat(y, yhat)

    def __p_of_y_given_y_hat(self,
                             y,
                             yhat):

        """
        y: np.ndarray, shape=(n1,n2,...,nK)
        phi: np.ndarray, shape=(n1,n2,...,nK)
        returns
        -------



        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.
            y: np.ndarray, shape=(n1,n2,...,nK

        yhat: (array-like of floats)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p: np.ndarray, shape=(n1,n2,...,nK)

        """

        check(self.regression_type=='GE', "This method works on with GE regression.")
        # Get GE noise model based on the users input.
        ge_noise_model = globals()[self.ge_noise_model_type + 'NoiseModel'](self, yhat, None)

        vec_p_of_y_given_yhat = np.vectorize(ge_noise_model.p_of_y_given_yhat)

        # store shape of return distribution

        return vec_p_of_y_given_yhat(y, yhat)


    def na_p_of_all_y_given_phi(self,
                                phi):

        """
        Evaluate the MPA measurement process at specified values of phi (the latent phenotype).

        parameters
        ----------

        phi: (array-like)
            Latent phenotype values at which to evaluate the measurement process.

        returns
        -------
        p_of_dot_given_phi: (array-like)
            Measurement process p(y|phi) for all possible values of y. Is of size
            MxY where M=len(phi) and Y is the number of possible y values.

        """

        check(self.regression_type == 'MPA', 'regression type must be "MPA" for this function ')

        p_of_dot_given_phi = self.model.p_of_all_y_given_phi(phi)

        return p_of_dot_given_phi


    # TODO: this function is possibly experiencing a tensorflow backend bug with a single example x, need to check.
    def p_of_y_given_x(self, y, x):

        """
        Method that computes p_of_y_given_x.

        parameters
        ----------
        y: (array-like of floats)
            The y values for which the conditional probability will be computed.

        x: (array-like of strings)
            The value on which the computed probability will be conditioned.

        returns
        -------
        p_of_y_given_x: (array-like of floats)
            Probability of y given sequence x. Shape of returned value will
            match shape of y_test.

        """


        if self.regression_type=='GE':
            yhat = self.x_to_yhat(x)
            # Get GE noise model based on the users input.
            ge_noise_model = globals()[self.ge_noise_model_type + 'NoiseModel'](self,yhat)

            p_of_y_given_x = ge_noise_model.p_of_y_given_yhat(y, yhat)
            return p_of_y_given_x

        elif self.regression_type=='MPA':

            # check that entered y (specifying bin number) is an integer
            check(isinstance(y, int),
                  'type(y), specifying bin number, must be of type int')

            # check that entered bin nnumber doesn't exceed max bins
            check(y< self.model.y_train[0].shape[0], "bin number cannot be larger than max bins = %d" %self.model.y_train[0].shape[0])


            phi = self.x_to_phi(x)
            p_of_y_given_x = self.p_of_y_given_phi(y, phi)
            return p_of_y_given_x

    def save(self,
             filename):

        """
        Method that will save the mave-nn model

        parameters
        ----------
        filename: (str)
            filename of the saved model.

        returns
        -------
        None

        """

        # save weights
        self.get_nn().save_weights(filename + '.h5')

        if self.regression_type=='GE':

            # get GE model configuration which will be used to reload the model
            GE_dict = self.__dict__.copy()

            # keep a single instance of training data, used for initalizing model.Model
            single_x = GE_dict['x'][0]
            single_y = GE_dict['y'][0]

            # remove all training data ...
            GE_dict.pop('x', None)
            GE_dict.pop('y', None)

            # retain a single training instance for quick loading
            GE_dict['x'] = single_x
            GE_dict['y'] = single_y

            # save these parameters to ensure
            GE_dict['diffeomorphic_mean'] = self.diffeomorphic_mean
            GE_dict['diffeomorphic_std'] = self.diffeomorphic_std

            # save model configuration
            pd.DataFrame(GE_dict, index=[0]).to_csv(filename+'.csv')
        else:
            NAR_dict = self.__dict__.copy()

            # store a single example
            single_x = NAR_dict['x'][0]
            #single_y = [NAR_dict['y'][0]]
            single_y = self.model.y_train[0]

            single_ct = [NAR_dict['ct_n'][0]]

            # remove training data ...
            NAR_dict.pop('x', None)
            NAR_dict.pop('y', None)
            NAR_dict.pop('ct_n', None)

            NAR_dict['diffeomorphic_mean'] = self.diffeomorphic_mean
            NAR_dict['diffeomorphic_std'] = self.diffeomorphic_std

            # and replace with single examples for quick loading
            NAR_dict['x'] = single_x
            NAR_dict['y'] = [single_y]
            NAR_dict['ct_n'] = [single_ct]

            pd.DataFrame(NAR_dict, index=[0]).to_csv(filename + '.csv')

    from mavenn.src.error_handling import check, handle_errors
    from mavenn.src.validate import validate_alphabet
    from mavenn.src.utils import get_1pt_variants


    @handle_errors
    def get_1pt_effects(self, wt_seq, out_format="matrix"):
        """
        Returns effects of all single-point mutations to a
        wild-type sequence in convenient formats.

        parameters
        ----------

        wt_seq: (str)
            The wild-type sequence.

        out_format: ("matrix" or "tidy")
            If matrix, a 2D matrix of dphi values is
            returned, with characters across columns and
            positions across rows. If "tidy", a tidy
            dataframe is returned that additionally lists
            all variant sequences, phi values, etc.

        returns
        -------

        out_df: (pd.DataFrame)
            Dataframe containing dphi values and other
            information.
        """

        # Get all 1pt variant sequences
        df = get_1pt_variants(wt_seq=wt_seq, alphabet=self.alphabet,
                              include_wt=True)
        x = df['seq'].values

        # Compute dphi values
        df['phi'] = self.x_to_phi(x)
        df['dphi'] = df['phi'] - df['phi']['WT']

        if out_format == "tidy":
            mut_df = df
        elif out_format == "matrix":
            # Keep only non-wt rows
            ix = (df.index != 'WT')
            tmp_df = df[ix]

            # Pivot matrix and return
            mut_df = tmp_df.pivot(index='pos', columns='mut_char',
                                  values='dphi')
            mut_df.fillna(0, inplace=True)
            mut_df.columns.name = None
        else:
            mut_df = None
            check(out_format in ["tidy", "matrix"],
                  f"out_format={out_format}; must be 'tidy' or 'matrix'.")

        return mut_df


    def get_additive_parameters(self, out_format="matrix"):
        """
        Returns additive parameters of to model in
        convenient formats.

        parameters
        ----------

        out_format: ("matrix" or "tidy")
            If matrix, a 2D matrix of dphi values is
            returned, with characters across columns and
            positions across rows. If "tidy", a tidy
            dataframe is returned that additionally lists
            all variant sequences, phi values, etc.

        returns
        -------

        out_df: (pd.DataFrame)
            Dataframe containing dphi values and other
            information.
        """
        param_df = self.get_gpmap_parameters()

        # Compile regular expression pattern
        pattern = re.compile('^theta_([0-9]+):([A-Z])$')

        # Remove non-additive parameters
        matches = [pattern.match(name) for name in param_df['name']]
        ix = [bool(m) for m in matches]
        param_df = param_df[ix]

        # Parse pos and char from parameter names
        matches = [pattern.match(name) for name in param_df['name']]
        param_df['pos'] = [int(m.group(1)) for m in matches]
        param_df['char'] = [m.group(2) for m in matches]

        if out_format == "tidy":
            out_df = param_df
        elif out_format == "matrix":
            out_df = param_df.pivot(index='pos', columns='char', values='value')
            out_df.columns.name = None
        else:
            out_df = None
            check(out_format in ["tidy", "matrix"],
                  f"out_format={out_format}; must be 'tidy' or 'matrix'.")

        # Return to user
        return out_df

