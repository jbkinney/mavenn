.. _totalinfo:

==========================================
``totalinfo``
==========================================

.. contents::

Overview
-------------
``totalinfo`` is a program within the mpathic package which calculates
the mutual information between gene expression and knowledge of the entire sequence.
This should be run on a sublibrary, which is a typical experiment, only run with 
only 200-300 unique sequences. The mutual information calculated represents the 
maximum amount of information acheivable by any model.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: totalinfo

   

   
Example Input and Output
-----------

Example input table::

    seq    ct_0    ct_1     ct_2
    ACGGT  2000    1500     3000
    AGGTT  100     4050     6500
    ...

Example Output Table::

    MI    std
    .805  .05



.. include:: weblinks.txt
