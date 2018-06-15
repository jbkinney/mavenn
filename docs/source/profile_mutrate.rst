==========================================
mpa.ProfileMut
==========================================

**Overview**

It is often useful to compute the mutation rate within a set of sequences, e.g., in order to
validate the composition of a library. This can be accomplished using the profile mut class as follows:


**Usage**

    >>> import mpathic as mpa
    >>> mpa.ProfileMut(dataset_df = valid_dataset)

**Example Input**::

    ct                     seq

    259  TAATGTGAGTTAGCTCACTCAT
    41  TAAAGTGAGTTAGCTCACTCAT
    36  TAATGTGAGTAAGCTCACTCAT
    35  TAGTGTGAGTTAGCTCACTCAT
    34  TAATGTTAGTTAGCTCACTCAT
    34  TTATGTGAGTTAGCTCACTCAT
    ...

**Example Output**::

        pos wt      mut

    0     0  T  0.23819
    1     1  A  0.24141
    2     2  A  0.24118
    3     3  T  0.24016
    4     4  G  0.24093
    5     5  T  0.24001
    ...

Class Details
-------------

.. autoclass:: profile_mut.ProfileMut
    :members: