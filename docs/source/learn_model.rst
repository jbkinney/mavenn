.. _learn_matrix:

==========================================
``learn_matrix``
==========================================

.. contents::

Overview
-------------
``learn_matrix`` is a program within the mpathic package which generates
linear energy matrix models for sections of a sorted library.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: learn_matrix

   

   
Example Input and Output
-----------

The input table to this program must contain a sequences column and counts columns
for each bin. For a sort seq experiment, this can be any number of bins. For MPRA
and selection experiments this must be ct_0 and ct_1.

Example Input Table::

   seq    ct_0     ct_1     ct_2    ...
   ACG    1        5        7
   GGT    8        5        5
   ...

Example Output Table::

   pos    val_A    val_C    val_G     val_T
   0      .04      -.3      -.2       .15
   1      .2       .1       -.44      .05
   ...




.. include:: weblinks.txt
