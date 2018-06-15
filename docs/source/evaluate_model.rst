==========================================
mpa.EvaluateModel
==========================================

.. contents::

Overview
-------------
``EvaluateModel`` can be used to predict the activity of arbitrary sequences.

Usage
---------------------

    >>> import mpathic as mpa
    >>> model = mpa.io.load_model("./mpathic/data/sortseq/full-0/crp_model.txt")
    >>> dataset = mpa.io.load_dataset("./mpathic/data/sortseq/full-0/data.txt")
    >>> mpa.EvaluateModel(dataset_df = dataset, model_df = model)


Example Input and Output
------------------------


Example Input Table::

    pos      val_A      val_C      val_G      val_T
    3  -0.070101  -0.056502   0.184170  -0.057568
    4  -0.045146  -0.042017   0.172377  -0.085214
    5  -0.035447   0.006974   0.059453  -0.030979
    6  -0.037837  -0.000299   0.079747  -0.041611
    7  -0.110627  -0.054740   0.066257   0.099110
    ...

Example Output Table::

   output:

    0        0.348108
    1       -0.248134
    2        0.009507
    3        0.238852
    4       -0.112121
    5       -0.048588
   ...


Class Details
-------------

.. autoclass:: evaluate_model.EvaluateModel
    :members:

