.. _profile_counts:

==========================================
``profile_counts``
==========================================

.. contents::

Overview
-------------
``profile_counts`` is a program within the mpathic package which tallies
the occurance of each base or amino acid at each position and outputs them in
a table.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: profile_counts

   

   
Example Input and Output
-----------

The input table must have at least a column for sequences, and a column with counts.

Example input table::

    seq       ct
    ACAGGT    10
    ACGGTT    9
    ...

Alternatively by using the --bin k option, another bin can be profiled.

Then the example table must at least have a column of sequences and a column labeled ct_k::

    seq       ct_1
    ACAGGT    10
    ACGGTT    9
    ...

An example command to run the analysis is ::
    
    sortseq profile_counts -i input_table.txt 




.. include:: weblinks.txt
