Implementation
==============

Testing
-------

.. autofunction:: mavenn.run_tests

Built-in examples
-----------------

.. autofunction:: mavenn.list_tutorials

.. autofunction:: mavenn.run_demo

.. autofunction:: mavenn.load_example_dataset

.. autofunction:: mavenn.load_example_model

Loading models
--------------

.. autofunction:: mavenn.load

Visualization
-------------

.. autofunction:: mavenn.heatmap

.. autofunction:: mavenn.heatmap_pairwise

Model class
-----------

.. autoclass:: mavenn.Model
    :members: set_data, fit, get_theta, get_nn,
        x_to_phi, phi_to_yhat, simulate_dataset, I_likelihood,
        I_predictive, yhat_to_yq, p_of_y_given_phi, p_of_y_given_yhat,
        save, p_of_y_given_x, x_to_yhat
