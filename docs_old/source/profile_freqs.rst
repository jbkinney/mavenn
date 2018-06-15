.. _profile_freqs:

==========================================
``profile_freqs``
==========================================

.. contents::

Overview
-------------
``profile_freqs`` is a program within the mpathic package which calculates the
fractional occurrence of each base or amino acid at each position.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: profile_freqs

   

   
Example Input and Output
-----------

Input tables must contain a position column (labeled ''pos'') and columns for
each base or amino acid (labeled ct_A, ct_C...).

Example Input Table::

    pos ct_A ct_C ct_G ct_T
    0   10   20   40   30
    ...

Example Output Table::

    pos freq_A freq_C freq_G freq_T
    0   .1     .2     .4     .3
    ...

An example command to run the analysis is ::

    sortseq profile_freqs -i counts_table.txt


.. include:: weblinks.txt
