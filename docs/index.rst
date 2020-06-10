========================================================================================
MAVENN: Quantitative Modeling of Sequence-Function Relationships using Neural Networks
========================================================================================

MAVE-NN is a python package for inferring models of sequence-function relationships from 
multiplexed assays of variant effects (MAVEs). MAVE-NN currently implements global epistasis 
regression and noise agnostic regression. Both models are implemented as neural networks 
using TensorFlow. The :ref:`installation`, :ref:`quickstart`,  :ref:`examples` 
sections below are provided to help users  quickly get  MAVE-NN working for 
their own research needs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. _installation:

Installation
--------------

MAVENN has minimal dependencies and is compatible with both Python 2.7 and Python 3.6.
The code for MAVENN is available on `GitHub <https://github.com/jbkinney/mavenn>`_ under an MIT open source license.
mavenn can be installed from `PyPI <https://pypi.org/project/logomaker/>`_ using the ``pip`` package manager by executing the following at the commandline: ::

    pip install mavenn

.. _quickstart:

Quick Start
-----------

For a quick demonstration of mavenn, execute the following within Python::

   import mavenn
   mavenn.demo(name='GEmpsa')


Resources
---------

.. toctree::
    :maxdepth: 2

    examples
    implementation


Reference
----------

Contact
-------

For technical assistance or to report bugs, please contact Ammar Tareen (`Email: tareen@cshl.edu <tareen@cshl.edu>`_, `Twitter: @AmmarTareen1 <https://twitter.com/AmmarTareen1>`_) . For more general correspondence, please contact Justin Kinney (`Email: jkinney@cshl.edu <jkinney@cshl.edu>`_, `Twitter: @jbkinney <https://twitter.com/jbkinney>`_).

Links
-----

- `Kinney Lab <http://kinneylab.labsites.cshl.edu/>`_
- `Cold Spring Harbor Laboratory <https://www.cshl.edu/>`_
