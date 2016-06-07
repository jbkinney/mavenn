.. _simulate_library:

==========================================
``simulate_library``
==========================================

.. contents::

Overview
-------------
``simulate_library`` is a program within the mpathic package which creates a library of
random mutants from an initial wildtype sequence and mutation rate.

After you install `mpathic`_, this program will be available to run at the command line. 

Command-line usage
---------------------
.. argparse::
   :module: sortseq.sortseq_for_doc
   :func: parser
   :prog: sortseq
   :path: simulate_library

   

   
Examples Inputs and outputs
-----------

To generate a library of mutated sequences you could use the command::
    
    sortseq simulate_library -w ACAGGGTTAC -n 50000 -m .2

This will output a simulated library of the form:: 

    seq           ct
    ACAGGGTTAC    100
    ACGGGGTTAC    50
    ...


     




.. include:: weblinks.txt
