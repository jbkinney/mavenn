.. _fromto_mutrate:

==========================================
``fromto_mutrate``
==========================================

.. contents::

Overview
-------------
``fromto_mutrate`` is a program which characterizes the biases in mutations within
your library. Specifically, given a wild type sequence. It calculates the probability
of occurance of all bases. 

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: fromto_mutrate

   

   
Example Input and Output
-----------
An unsorted library (or a sorted library with an unsorted counts bin ''ct'') must
be used as input.

Example Input Table::

   seq    ct
   ACA    9
   GGG    7
   ...

Example Output Table::

   wt   obs   mut
   A    A     .97
   A    C     .01
   A    G     .01
   ...

The final column ''mut'' gives the probability that we will see the base in the
column ''obs'' given the wt base is what is shown in the column ''wt''.

The analysis could be performed using the example command::

    sortseq fromto_mutrate -i unsorted_lib.txt -o my_mutrates.txt


.. include:: weblinks.txt
