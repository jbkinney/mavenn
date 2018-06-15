==========================================
mpa.ProfileInfo
==========================================

**Overview**

``profile_info`` is a program within the mpathic package which calculates
the mutual information between base identity at a given position and expression
for each position in the given data set.

**Usage**

    >>> import mpathic as mpa
    >>> mpa.ProfileInfo(dataset_df = dataset_df)
   
**Example Input and Output**

The input to the function must be a sorted library a column for sequences and 
columns of counts for each bin. For selection experiments, ct_0 should label the
pre-selection library and ct_1 should be the post selection library. For MPRA
experiments, ct_0 should label the sequence library counts, and ct_1 should
label the mRNA counts.

**Example input table**::

    seq       ct_0    ct_1     ct_2...
    ACATT     1       4        3
    GGATT     2       5        5
    ...

**Example output table**::

    pos    info    info_err
    0      .02     .004
    1      .04     .004
    ...

The mutual information is given in bits.


Class Details
-------------

.. autoclass:: profile_info.ProfileInfo
    :members:

