"""
inpyt_layer.py: Defines the input layer to a mavenn model.
TODO: need to add additional args to __init__ of InputLayer to allow for custom features.
"""

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors

# tensorflow importants
from tensorflow.keras.layers import Input, Lambda


class InputLayer:

    """
    number_x_nodes: (int)
    Number of inputs nodes to a mavenn model.
    """

    @handle_errors
    def __init__(self,
                 number_x_nodes,
                 number_of_targets=1):

        self.number_x_nodes = number_x_nodes
        self.number_of_targets = number_of_targets

    def get_input_layer(self):

        # need to add one additional node for targets, which will be used
        # calcalate likelihood (i.e., using x and y)
        # Note that if additional input features are to be added (e.g., shape),
        # the following code will have to be updated.
        number_input_layer_nodes = self.number_x_nodes + self.number_of_targets

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:self.number_x_nodes],
                                output_shape=((self.number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, self.number_x_nodes:self.number_x_nodes + 1],
            output_shape=((self.number_of_targets,)),
            trainable=False, name='Labels_input')(inputTensor)

        return inputTensor, sequence_input, labels_input


