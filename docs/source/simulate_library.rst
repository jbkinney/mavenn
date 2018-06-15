==========================================
mpa.SimulateLibrary
==========================================

Overview
--------
simulate library is a program within the mpathic package which creates a library of
random mutants from an initial wildtype sequence and mutation rate.


Usage
-----

    >>> import mpathic
    >>> mpathic.SimulateLibrary(wtseq="TAATGTGAGTTAGCTCACTCAT")


**Example Output Table**::

    ct            seq
    1002          TAATGTGAGTTAGCTCACTCAT
    50            TAATGTGAGTTAGATCACTCAT
    ...


Class Details
-------------

.. autoclass:: simulate_library.SimulateLibrary
    :members: 
