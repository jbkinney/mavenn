.. _profile_enrichment:

==========================================
``profile_enrichment``
==========================================

.. contents::

Overview
-------------
``profile_enrichment`` is a program within the mpathic package which calculates
the log enrichment of each amino acid at each position. Pseudo counts are added to
each entry during the calculation.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: profile_enrichment

   

   
Example Input and Output
-----------

Input tables must be a table containing a column for sequences, counts before selection
and counts after selection.

Example Input Table::


    seq    ct_0    ct_1
    GAPY   10      34
    APFY   30      10
    ...

Example Output Table::

    pos   le_A    le_C ...
    0     .1      -.2
    1     .4      -.3
    ...

The analysis can be run using the command::

   sortseq profile_enrichment -i selected_library.txt

Where selected_library.txt is the file containing your input table.



.. include:: weblinks.txt
