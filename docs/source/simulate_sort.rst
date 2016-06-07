.. _simulate_sort:

==========================================
``simulate_sort``
==========================================

.. contents::

Overview
-------------
``simulate_sort`` is a program within the mpathic package which simulates
performing a Sort Seq experiment.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: simulate_sort

   

   
Example Input and Output
-----------

The input table to this function must contain sequence, counts, and energy columns

Example Input Table::

   seq    ct    val
   AGGTA  5     -.4
   AGTTA  1     -.2
   ...

Example Output Table::

   seq    ct    val    ct_1     ct_2     ct_3 ...
   AGGTA  5     -.4    1        2        1
   AGTTA  1     -.2    0        1        0
   ...

The output table will contain all the original columns, along with the sorted columns (ct_1, ct_2 ...)

An example command to execute this analysis::

    sortseq simulate_sort -i my_library.txt -nm LogNormal -o my_sorted.txt


.. include:: weblinks.txt
