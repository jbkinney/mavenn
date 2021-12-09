=================================================================================
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
=================================================================================

MAVE-NN [#Tareen2020]_ enables the rapid quantitative modeling of genotype-phenotype (G-P) maps from the data produced by multiplex assays of variant effect (MAVEs). Such assays include deep mutational scanning (DMS) experiments on proteins, massively parallel reporter assays (MPRAs) on DNA or RNA regulatory sequences, and more. MAVE-NN conceptualizes G-P map inference as a problem in information compression; this problem is then solved by training a neural network using a TensorFlow backend. To learn more about this modeling strategy, please see our bioRxiv preprint.

.. [#Tareen2020] Tareen A, Kooshkbaghi M, Posfai A, Ireland WT,  McCandlish DM, Kinney JB.
    MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
    Biorxiv (2020). `<https://doi.org/10.1101/2020.07.14.201475>`_

MAVE-NN is written for Python 3 and is provided under an MIT open source license. The documentation provided here is meant help users quickly get MAVE-NN working for their own research needs. Please do not hesitate to contact us with any questions or suggestions for improvements. For technical assistance or to report bugs, please contact Ammar Tareen (`Email: tareen@cshl.edu <tareen@cshl.edu>`_, `Twitter: @AmmarTareen1 <https://twitter.com/AmmarTareen1>`_) . For more general correspondence, please contact Justin Kinney (`Email: jkinney@cshl.edu <jkinney@cshl.edu>`_, `Twitter: @jbkinney <https://twitter.com/jbkinney>`_).


Installation
-----------------

MAVE-NN can be installed from `PyPI <https://pypi.org/project/mavenn/>`_
using the ``pip`` package manager by executing the following at the
commandline: ::

    $ pip install mavenn

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


Modeling tutorials
------------------

MAVE-NN comes with a variety of pre-trained models that users can load and apply. Models similar to these can be trained using the following notebooks

.. toctree::
    :maxdepth: 1

    tutorials/1_demos.ipynb
    tutorials/2_protein_dms_additive_gpmaps.ipynb
    tutorials/3_splicing_mpra_multiple_gpmaps.ipynb
    tutorials/4_protein_dms_biohysical_gpmap.ipynb

Built-in datasets
-----------------

MAVE-NN provides multiple built-in datasets that users can easily load and use to train their own models

.. toctree::
    :maxdepth: 1

    datasets/overview
    datasets/dataset_gb1
    datasets/dataset_amyloid
    datasets/dataset_tdp43
    datasets/dataset_mpsa
    datasets/dataset_sortseq


Documentation
-------------

.. toctree::
    :maxdepth: 1

    methods
    implementation

Links
-----

- `Kinney Lab <http://kinneylab.labsites.cshl.edu/>`_
- `Cold Spring Harbor Laboratory <https://www.cshl.edu/>`_

