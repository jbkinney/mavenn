==========================================
mpa.ProfileFreq
==========================================

**Overview**

``ProfileFreq`` is a program within the mpathic package which calculates the
fractional occurrence of each base or amino acid at each position.


**Usage**

    >>> import mpathic as mpa
    >>> mpa.ProfileFreq(dataset_df = dataset_df)
   
   
**Example Input and Output**

Input tables must contain a position column (labeled ''pos'') and columns for
each base or amino acid (labeled ct_A, ct_C...).

**Example Input Table**::

    pos ct_A ct_C ct_G ct_T
    0   10   20   40   30
    ...

**Example Output Table**::

    pos freq_A freq_C freq_G freq_T
    0   .1     .2     .4     .3
    ...


Class Details
-------------

.. autoclass:: profile_freq.ProfileFreq
    :members: