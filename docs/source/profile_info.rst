.. _profile_info:

==========================================
``profile_info``
==========================================

.. contents::

Overview
-------------
``profile_info`` is a program within the mpathic package which calculates
the mutual information between base identity at a given position and expression
for each position in the given data set.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: profile_info

   

   
Example Input and Output
-----------

The input to the function must be a sorted library a column for sequences and 
columns of counts for each bin. For selection experiments, ct_0 should label the
pre-selection library and ct_1 should be the post selection library. For MPRA
experiments, ct_0 should label the sequence library counts, and ct_1 should
label the mRNA counts.

Example input table::

    seq       ct_0    ct_1     ct_2...
    ACATT     1       4        3
    GGATT     2       5        5
    ...

Example output table::

    pos    info    info_err
    0      .02     .004
    1      .04     .004
    ...

The mutual information is given in bits.

An example command to run this analysis is::

    sortseq profile_info -i sorted_library.txt -s 20 -e 80 -o info_profile.txt    



.. include:: weblinks.txt
