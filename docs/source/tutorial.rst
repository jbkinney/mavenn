==========================================
Tutorial
==========================================

Import the MPAthic package as follows::

    import mpathic as mpa

Simulating Data
~~~~~~~~~~~~~~~

We begin by simulating a library of variant CRP binding sites. We can use the :doc:`simulate_library` class to
create a library of random mutants from an initial wildtype sequence and mutation rate::

    sim_library = mpa.SimulateLibrary(wtseq="TAATGTGAGTTAGCTCACTCAT", mutrate=0.24)
    sim_library.output_df.head()

The `output_df` attribute of the ``sim_library`` class looks like the dataframe below

+------------------+------------------------------+
|      ct          | seq                          |
+==================+==============================+
|      21          | TAATGTGAGTTAGCTCACTCAT       |
+------------------+------------------------------+
|      7           | TAATGTGAGTTAGCTAACTCAT       |
+------------------+------------------------------+
|      6           | TAATGTGAGTTAGCTCACTCAA       |
+------------------+------------------------------+

⋮

+------------------+------------------------------+
|      1           | TAATGTGTGTTCGCTCATCCAT       |
+------------------+------------------------------+


In general, MPAthic datasets are pandas dataframes, comprising of columns of counts and sequence values. To simulate
a Sort-Seq experiment ([#Kinney2010]_), we use the :doc:`simulate_sort` class. This class requires a dataset input
and a model dataframe input. We first import these inputs using ``io`` module provided with the MPAthic package::

    # Load dataset and model dataframes
    dataset_df = mpa.io.load_dataset('sort_seq_data.txt')
    model_df = mpa.io.load_model('crp_model.txt')

Next, we call the ``SimulateSort`` class as follows::

    # Simulate a Sort-Seq experiment
    sim_sort = mpa.SimulateSort(df=dataset_df,mp=model_df)
    sim_sort.output_df.head()

The head of the output dataframe looks like

+----+------+------+------+------+------+------------------------+
| ct | ct_0 | ct_1 | ct_2 | ct_3 | ct_4 | seq                    |
+====+======+======+======+======+======+========================+
| 4  | 0    | 0    | 1    | 0    | 0    | AAAAAAGGTGAGTTAGCTAACT |
+----+------+------+------+------+------+------------------------+
| 3  | 0    | 0    | 0    | 0    | 1    | AAAAAATATAAGTTAGCTCGCT |
+----+------+------+------+------+------+------------------------+
| 4  | 0    | 0    | 0    | 1    | 0    | AAAAAATATGATTTAGCTGACT |
+----+------+------+------+------+------+------------------------+
| 3  | 0    | 0    | 0    | 0    | 1    | AAAAAATGTCAGTTAGCTCACT |
+----+------+------+------+------+------+------------------------+
| 4  | 0    | 0    | 1    | 0    | 0    | AAAAAATGTGAATTATCGCACT |
+----+------+------+------+------+------+------------------------+

Computing Profiles
~~~~~~~~~~~~~~~~~~

It is often useful to compute the mutation rate within a set of sequences, e.g., in order to validate the
composition of a library. This can be accomplished using the :doc:`profile_mutrate` class as follows::

   profile_mut = mpa.ProfileMut(dataset_df = dataset_df)
   profile_mut.mut_df.head()

The mutation rate at each position within the sequences looks like

+-----+----+----------+
| pos | wt | mut      |
+=====+====+==========+
| 0   | A  | 0        |
+-----+----+----------+
| 1   | A  | 0        |
+-----+----+----------+
| 2   | A  | 0.33871  |
+-----+----+----------+
| 3   | T  | 0.127566 |
+-----+----+----------+
| 4   | A  | 0.082111 |
+-----+----+----------+


To view the frequency of occurrence for every base at each position, use the :doc:`profile_freq` class::

   profile_freq = mpa.ProfileFreq(dataset_df = dataset_df)
   profile_freq.freq_df.head()

+-----+----------+----------+----------+----------+
| pos | freq_A   | freq_C   | freq_G   | freq_T   |
+=====+==========+==========+==========+==========+
| 0   | 1        | 0        | 0        | 0        |
+-----+----------+----------+----------+----------+
| 1   | 1        | 0        | 0        | 0        |
+-----+----------+----------+----------+----------+
| 2   | 0.66129  | 0.33871  | 0        | 0        |
+-----+----------+----------+----------+----------+
| 3   | 0.043988 | 0.042522 | 0.041056 | 0.872434 |
+-----+----------+----------+----------+----------+
| 4   | 0.917889 | 0.019062 | 0.02566  | 0.03739  |
+-----+----------+----------+----------+----------+

Information proles (also called "information footprints") provide a particularly useful way to identify
functional positions within a sequence. These proles list, for each position in a sequence, the mutual
information between the character at that position and the bin in which a sequence is found. Unlike mutation
and frequency profiles, which require sequence counts for a single bin only, information profiles are
computed from full datasets, and can be accomplished using the :doc:`profile_info` class as follows::

   profile_info = mpa.ProfileInfo(dataset_df = dataset_df)
   profile_info.info_df.head()

+-----+----------+
| pos | info     |
+=====+==========+
| 0   | 0.000077 |
+-----+----------+
| 1   | 0.000077 |
+-----+----------+
| 2   | 0.008357 |
+-----+----------+
| 3   | 0.008743 |
+-----+----------+
| 4   | 0.013745 |
+-----+----------+

Quantitative Modeling
~~~~~~~~~~~~~~~~~~~~~~

The :doc:`learn_model` class can be used to fit quantitative models to data::

   learned_model = mpa.LearnModel(df=dataset_df)
   learned_model.output_df.head()

+-----+-----------+-----------+----------+----------+
| pos | val_A     | val_C     | val_G    | val_T    |
+=====+===========+===========+==========+==========+
| 0   | -0.201587 | 0.067196  | 0.067196 | 0.067196 |
+-----+-----------+-----------+----------+----------+
| 1   | -0.201587 | 0.067196  | 0.067196 | 0.067196 |
+-----+-----------+-----------+----------+----------+
| 2   | -0.10637  | -0.167351 | 0.13686  | 0.13686  |
+-----+-----------+-----------+----------+----------+
| 3   | -0.287282 | 0.041222  | -0.2039  | 0.44996  |
+-----+-----------+-----------+----------+----------+
| 4   | -0.056109 | -0.871858 | 0.344537 | 0.583429 |
+-----+-----------+-----------+----------+----------+

The purpose of having a quantitative model is to be able to predict the activity
of arbitrary sequences. This basic operation is accomplished using the :doc:`evaluate_model` class::

   eval_model = mpa.EvaluateModel(dataset_df = dataset_df, model_df = model_df)
   eval_model.out_df.head()

+----+------+------+------+------+------+------------------------+-----------+
| ct | ct_0 | ct_1 | ct_2 | ct_3 | ct_4 | seq                    | val       |
+====+======+======+======+======+======+========================+===========+
| 1  | 0    | 0    | 1    | 0    | 0    | AAAGGTGAGTTAGCTAACTCAT | 0.348108  |
+----+------+------+------+------+------+------------------------+-----------+
| 1  | 0    | 0    | 0    | 0    | 1    | AAATATAAGTTAGCTCGCTCAT | -0.248134 |
+----+------+------+------+------+------+------------------------+-----------+
| 1  | 0    | 0    | 0    | 1    | 0    | AAATATGATTTAGCTGACTCAT | 0.009507  |
+----+------+------+------+------+------+------------------------+-----------+
| 1  | 0    | 0    | 0    | 0    | 1    | AAATGTCAGTTAGCTCACTCAT | 0.238852  |
+----+------+------+------+------+------+------------------------+-----------+
| 1  | 0    | 0    | 1    | 0    | 0    | AAATGTGAATTATCGCACTCAT | -0.112121 |
+----+------+------+------+------+------+------------------------+-----------+

Often, it is useful to scan a model over all sequences embedded within larger contigs. To
do this, MPAthic provides the class :doc:`scan_model`, which is called as follows::

    # get contigs, provided with mpathic
    fastafile = "./mpathic/examples/genome_ecoli_1000lines.fa"
    contig = mpa.io.load_contigs_from_fasta(fastafile, model_df)

    scanned_model = mpa.ScanModel(model_df = model_df, contigs_list = contigs_list)
    scanned_model.sitelist_df.head()

+---+----------+------------------------+-------+-------+-----+-----------+
|   | val      | seq                    | left  | right | ori | contig    |
+===+==========+========================+=======+=======+=====+===========+
| 0 | 2.040628 | GGTCGTTTGCCTGCGCCGTGCA | 11710 | 11731 | +   | MG1655.fa |
+---+----------+------------------------+-------+-------+-----+-----------+
| 1 | 2.00608  | GGAAGTCGCCGCCCGCACCGCT | 74727 | 74748 | -   | MG1655.fa |
+---+----------+------------------------+-------+-------+-----+-----------+
| 2 | 1.996992 | TGGGTGTGGCGCGTGACCTGTT | 45329 | 45350 | +   | MG1655.fa |
+---+----------+------------------------+-------+-------+-----+-----------+
| 3 | 1.920821 | GGTATGTGTCGCCAGCCAGGCA | 38203 | 38224 | +   | MG1655.fa |
+---+----------+------------------------+-------+-------+-----+-----------+
| 4 | 1.879852 | GGTGATTTTGGCGTGGTGGCGT | 73077 | 73098 | -   | MG1655.fa |
+---+----------+------------------------+-------+-------+-----+-----------+

A good way to assess the quality of a model is to compute its predictive information on a massively
parallel data set. This can be done using the `predictive_info` (need to write this) class::

   predictive_info = mpa.PredictiveInfo(data_df = dataset_df, model_df = model_df,start=52)

**References**


.. [#Kinney2010] Kinney JB, Anand Murugan, Curtis G. Callan Jr., and Edward C. Cox (2010) `Using deep sequencing to characterize the biophysical mechanism of a transcriptional regulatory sequence. <http://www.pnas.org/content/107/20/9158>`_ PNAS May 18, 2010. 107 (20) 9158-9163;
   :download:`PDF <Kinney2010.pdf>`.

