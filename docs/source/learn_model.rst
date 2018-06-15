==========================================
mpa.LearnModel
==========================================

.. contents::

Overview
-------------
``LearnModel`` is a program within the mpathic package which generates
linear energy matrix models for sections of a sorted library.


**Usage**::

    >>> import mpathic
    >>> loader = mpathic.io
    >>> filename = "./mpathic/data/sortseq/full-0/data.txt"
    >>> df = loader.load_dataset(filename)
    >>> mpathic.LearnModel(df=df,verbose=True,lm='ER')


Example Input and Output
-------------------------

There are two types of input dataframes learn model can accept as input: Matrix models and neighbour models.
The input table to this program must contain a sequences column and counts columns
for each bin. For a sort seq experiment, this can be any number of bins. For MPRA
and selection experiments this must be ct_0 and ct_1.

**Matrix models Input Dataframe**::


    seq       ct_0       ct_1       ct_2       ct_3       ct_4

    AAAAAAGGTGAGTTA   0.000000   0.000000   1.000000   0.000000   0.000000
    AAAAAATATAAGTTA   0.000000   0.000000   0.000000   0.000000   1.000000
    AAAAAATATGATTTA   0.000000   0.000000   0.000000   1.000000   0.000000
    ...

**Neighbour Model**::


       pos     val_AA     val_AC     val_AG     val_AT     val_CA     val_CC     val_CG     val_CT     val_GA     val_GC     val_GG     val_GT     val_TA     val_TC     val_TG     val_TT
         0   0.081588  -0.019021   0.007188   0.042818  -0.048443  -0.015712  -0.053949  -0.024360  -0.025149  -0.030791  -0.022920  -0.026910   0.052324   0.002189  -0.014354   0.095505
         1   0.033288  -0.005410   0.014198   0.018246  -0.033583  -0.001761  -0.020431  -0.007561  -0.018550  -0.025738  -0.028961  -0.010787   0.007764   0.024888  -0.000199   0.054599
         2  -0.026142   0.008002  -0.029641   0.036698  -0.001028  -0.008025  -0.022645   0.023678   0.006907  -0.016295  -0.054918   0.028913  -0.005400   0.003121   0.000996   0.055780
         3  -0.046159  -0.006071  -0.001542   0.028109  -0.020442  -0.024574   0.056595  -0.024776  -0.005172  -0.055010  -0.029327  -0.016699   0.001295  -0.016304   0.128112   0.031967
        ...

**Example Output Table**::

    pos     val_A     val_C     val_G     val_T
    0     0  0.000831 -0.014006  0.144818 -0.131643
    1     1 -0.033734  0.087419 -0.029997 -0.023688
    2     2  0.009189  0.018999  0.026719 -0.054908
    3     3 -0.003516  0.073503  0.001759 -0.071745
    4     4  0.062168 -0.028879 -0.057249  0.023961
    ...



Class Details
-------------

.. autoclass:: learn_model.LearnModel
    :members:

