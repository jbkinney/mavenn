.. _simulate_sublibrary:

==========================================
``simulate_sublibrary``
==========================================

.. contents::

Overview
-------------
``simulate_sublibrary`` is a program which takes a library file and generates
a sublibrary file. A sublibrary file has only a small number of unique sequences
(200-300 usually) and many copies of each sequence. The sublibrary is useful
for using as a test data set to test the performance of your model. It is also 
useful for using totalinfo to calculate the best possible performance of any model.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: simulate_sublibrary

   

   
Example Input and Output
-----------

The input should be a sequence library.

Example Input Table::

    seq    ct
    ATTAG  1
    ACCTA  15
    GGATT  9
    ...

Example Output Table::

    seq    ct
    ATTAG  6000
    AGGAT  6000
    ...

By default, each chosen sequence is given a uniform number of counts. 

Example command to perform the analysis::

    sortseq simulate_sublibrary -i my_library.txt -o my_sublibrary.txt


.. include:: weblinks.txt
