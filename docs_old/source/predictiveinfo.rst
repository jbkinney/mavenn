.. _predictiveinfo:

==========================================
``predictiveinfo``
==========================================

.. contents::

Overview
-------------
``predictiveinfo`` is a program within the mpathic package which computes
the mutual information between a linear energy matrix model and a data set.
Models which are closer to reality will have a higher mutual information value.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: predictiveinfo

   

   
Example Input and Output
-----------

The input table should be a sorted library data set (the model should be specified
after the -m flag). You should use the --start and --end flags to specify the
region in the data set that the model corresponds to.

Example Input Table::

    seq    ct_0    ct_1  ...
    AGTT   20      13
    CCTA   35      40
    ...

Example Output Table::

    info
    .94

Example command to run the analysis::

   sortseq predictive info -i my_dataset.txt -m my_linear_matrix_model.txt -s 5 -e 20


.. include:: weblinks.txt
