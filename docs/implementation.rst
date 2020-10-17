Implementation
==============

Tests
-----

A suite of automated tests are provided to ensure proper software installation
and execution.

.. autofunction:: mavenn.run_tests

Examples
--------

A variety of real-world datasets, pre-trained models, analysis demos, and
tutorials can be accessed using the following functions.

.. autofunction:: mavenn.load_example_dataset

.. autofunction:: mavenn.load_example_model

.. autofunction:: mavenn.run_demo

.. autofunction:: mavenn.list_tutorials

Load
----

MAVE-NN allows users to save and load trained models.

.. autofunction:: mavenn.load

Visualization
-------------

MAVE-NN provides the following two methods to facilitate the visualization of
inferred genotype-phenotype maps.

.. autofunction:: mavenn.heatmap

.. autofunction:: mavenn.heatmap_pairwise

Models
------

The ``mavenn.Model`` class represents all neural-network-based models inferred
by MAVE-NN. A variety of class methods make it easy to,

    - define models,
    - fit models to data,
    - access model parameters and metadata,
    - save models,
    - evaluate models on new data.

In particular, these methods allow users to train and analyze models without
prior knowledge of TensorFlow 2, the deep learning framework used by MAVE-NN
as a backend.

.. autoclass:: mavenn.Model
    :members: set_data, fit, get_theta, get_nn,
        x_to_phi, phi_to_yhat, simulate_dataset, I_likelihood,
        I_predictive, yhat_to_yq, p_of_y_given_phi, p_of_y_given_yhat,
        save, p_of_y_given_x, x_to_yhat
