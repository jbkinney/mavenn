=================================================================================
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
=================================================================================

Version: |version| (Built on |today|)

MAVE-NN [#Tareen2020]_ enables the rapid quantitative modeling of genotype-phenotype (G-P) maps from the data produced by multiplex assays of variant effect (MAVEs). Such assays include deep mutational scanning (DMS) experiments on proteins, massively parallel reporter assays (MPRAs) on DNA or RNA regulatory sequences, and more. MAVE-NN conceptualizes G-P map inference as a problem in information compression; this problem is then solved by training a neural network using a TensorFlow backend. To learn more about this modeling strategy, please see our manuscript in Genome Biology.

.. [#Tareen2020] Tareen A, Kooshkbaghi M, Posfai A, Ireland WT,  McCandlish DM, Kinney JB.
    MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect
    Genome Biology (2022). `<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02661-7>`_

MAVE-NN is written for Python 3 and is provided under an MIT open source license. The documentation provided here is meant to help users quickly get MAVE-NN working for their own research needs. Please do not hesitate to contact us with any questions or suggestions for improvements. For technical assistance or to report bugs, please create an issue on the `MAVE-NN GitHub repository <https://github.com/jbkinney/mavenn>`_. For more general correspondence, please contact Justin Kinney (`Email: jkinney@cshl.edu <jkinney@cshl.edu>`_).

.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    installation
    tutorials
    datasets
    math
    api

Links
==============

- `Kinney Lab <http://kinneylab.labsites.cshl.edu/>`_
- `Cold Spring Harbor Laboratory <https://www.cshl.edu/>`_
