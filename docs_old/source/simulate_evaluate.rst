.. _simulate_evaluate:

==========================================
``simulate_evaluate``
==========================================

.. contents::

Overview
-------------
``simulate_evaluate`` is a program within the mpathic package which 
uses an energy model to calculate the predicted energies of each sequence in a
library. A simulated library must be evaluated before simulate_sort can be used.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: simulate_evaluate

   

   
Examples
-----------

The input table to this function should be a library file. The model type needs
to be specified using the -m flag. The file name of the model should be specified
after the -mp flag.

Example Input Table::

   seq    ct
   ACTAG  10
   AGGTA  5
   ...

Example Output Table::

   seq    ct     val
   ACTAG  10     -1.4
   AGGTA  5      -.5
   ...



.. include:: weblinks.txt
