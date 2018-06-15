.. _simulate_selection:

==========================================
``simulate_selection``
==========================================

.. contents::

Overview
-------------
``simulate_selection`` is a program within the mpathic package which simulates
a protein selection experiment.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: simulate_selection

   

   
Example Input and Output
-----------

Example Input Table::

    seq   ct    val
    ACY   10    -.8
    YTA   5     -.1
    ...

Example Output Table::

    seq   ct    val    ct_0    ct_1
    ACY   10    -.8    7       9
    YTA   5     -.1    8       1
    ...

Example Command to run the analysis::

   sortseq simulate_selection -i my_library.txt



.. include:: weblinks.txt
