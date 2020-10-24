.. _installation:

Installation
============

Requirements
------------

MAVE-NN runs in Python 3 using a TensorFlow 2 backend.

From PyPI
---------

MAVE-NN can be installed from `PyPI <https://pypi.org/project/mavenn/>`_
using the ``pip`` package manager by executing the following at the
commandline: ::

    $ pip install mavenn

From GitHub
-----------

Alternatively, you can clone MAVE-NN from
`GitHub <https://github.com/jbkinney/mavenn>`_ by doing
this at the command line: ::

    $ cd appropriate_directory
    $ git clone https://github.com/jbkinney/mavenn.git

where ``appropriate_directory`` is the absolute path to where you would like
MAVE-NN to reside. Then add this to the top of any Python file in
which you use MAVE-NN: ::

    # Insert local path to MAVE-NN at beginning of Python's path
    import sys
    sys.path.insert(0, 'appropriate_directory/mavenn')

    #Load mavenn
    import mavenn

Quickstart
----------

For a quick demonstration of MAVE-NN's capabilities, execute the following
within Python::

    import mavenn
    mavenn.run_demo('mpsa_ge_training', print_code=False)

This trains a model on data from a massively parallel splicing assay (MPSA)
performed by Wong et al. 2018 [#Wong2018]_. In our experience this takes approximately 30
seconds to complete on a standard laptop computer.
It produces the following figure, which illustrates model performance
and training history. If you set ``print_code=True``, the code used to perform
these computations will also be printed.

.. image:: _static/mpsa_ge_training.png

References
----------

.. [#Wong2018] Wong MS, Kinney JB, Krainer AR (2018). Quantitative activity profile and context dependence of all human 5'
    splice sites. `Mol Cell. 71(6):1012-1026.e3. <https://doi.org/10.1016/j.molcel.2018.07.033>`_