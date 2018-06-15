==========================================
mpa.ScanModel
==========================================

**Overview**

The scan model class can scan a model over all sequences embedde within larger contigs.

**Usage**

    >>> import mpathic as mpa
    >>> model = mpa.io.load_model("./mpathic/data/sortseq/full-0/crp_model.txt")
    >>> fastafile = "./mpathic/examples/genome_ecoli_1000lines.fa"
    >>> contig = mpa.io.load_contigs_from_fasta(fastafile,model)
    >>> mpa.ScanModel(model_df = model, contig_list = contig)

**Example Output Table**::


        val                     seq   left  right ori     contig
    0  2.040628  GGTCGTTTGCCTGCGCCGTGCA  11710  11731   +  MG1655.fa
    1  2.006080  GGAAGTCGCCGCCCGCACCGCT  74727  74748   -  MG1655.fa
    2  1.996992  TGGGTGTGGCGCGTGACCTGTT  45329  45350   +  MG1655.fa
    3  1.920821  GGTATGTGTCGCCAGCCAGGCA  38203  38224   +  MG1655.fa
    4  1.879852  GGTGATTTTGGCGTGGTGGCGT  73077  73098   -  MG1655.fa
    5  1.866188  GTTCTTTTCCGCGGGCTGGGAT  35967  35988   -  MG1655.fa
    ...

Class Details
-------------

.. autoclass:: scan_model.ScanModel
    :members:
