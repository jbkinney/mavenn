MPAthic 
========
*Quantitative modeling of sequence-function relationships for massively parallel assays*

Written by William T. Ireland and Justin B. Kinney  
Current version: 0.01.01  

Notice:   
This software package was formerly known as "**Sort-Seq Tools**"

Citation:  
Ireland WT, Kinney JB (2016) *MPAthic: quantitative modeling of sequence-function relationships for massively parallel assays.* bioRxiv doi:10.1101/054676

v0.01.01 Snapshot:    
DOI 10.5281/zenodo.55837    
https://zenodo.org/badge/latestdoi/22771/jbkinney/mpathic     

Please address questions or problems regarding this software to Justin B. Kinney at jkinney@cshl.edu.

## Overview

MPAthic is a software package for analyzing data from a variety massively parallel assays, including Sort-Seq assays, Massively Parallel Reporter assays, and Deep Mutational Scanning assays. MPAthic provides a set of command line routines, which are listed in the [documentation][documentation]. Details can be found in the [accompanying preprint][preprint].

## Requriements

MPAthic is written in Python 2.9.7. It has been verified to work on Linux and Mac OS X. Installation currently requires a number of other Python packages:
* biopython>=1.6
* pymc>=2.3.4
* scikit-learn>=0.15.2, <= 0.16.1
* statsmodels>=0.5.0
* mpmath>=0.19
* pandas>=0.16.0
* weblogo>=3.4
* Cython>=0.23.4
* matplotlib<=1.5.0

## Installation

To install MPAthic, clone this repository, navigate to the folder containing this README file, and execute

```
python setup.py install
```

Alternatively, MPAthic can be installed from PyPI by executing

```
pip install mpathic
```

This approach will (at least attempt to) install all of MPAthic's dependencies, as well as MPAthic itself. After MPAthic is installed, you can test the functionality of all methods by running

```
mpathic run_tests
```

This suite of tests takes ~10 min to execute. 

## Documentation

The commands used to perform the analysis in Ireland & Kinney (2016) are described in [analysis.md](analysis.md). Documentation on each of the MPAthic functions is provided [here][documentation].

## Quick start guide

Below are the commands described in the "Overview" section of the [Supplemental Information of Ireland and Kinney (2016)](http://biorxiv.org/content/early/2016/05/21/054676.figures-only). These commands provide a quick entry into the capabilities of MPAthic. To execute them, first change to the [examples](examples/) directory, which contains the necessary inpupt files [true_model.txt](examples/true_model.txt) and [genome_ecoli.fa](examples/genome_ecoli.fa). 

#### Simulating data

Simualte a library of binding sites for the CRP transcription factor:
```
mpathic simulate_library -w TAATGTGAGTTAGCTCACTCAT -n 100000 -m 0.24 -o library.txt
```

Simulate a Sort-Seq experiment using a model ([true_model.txt](examples/true_model.txt)) of CRP-DNA affinity:
```
mpathic simulate_sort -m true_model.txt -n 4 -i library.txt -o dataset.txt
```

#### Computing summary statistics

Compute a mutation profile of the simluated library:
```
mpathic profile_mut -i library.txt -o mutprofile.txt
```

Compute the occurance frequency of each base at each position in the library:
```
mpathic profile_freq -i library.txt -o freqprofile.txt
```

Compute an information profile (a.k.a information footprint) from the simulated data:
```
mpathic profile_info --err -i dataset.txt -o infoprofile.txt
```

#### Inferring quantitative models

Infer a matrix model for CRP from the simulated data:
```
mpathic learn_model -lm LS -mt MAT -i dataset.txt -o matrix_model.txt
```

Infer a neighbor model for CRP from the simulated data:
```
mpathic learn_model -lm LS -mt NBR -i dataset.txt -o neighbor_model.txt
```

#### Evaluating models

Evaluate the inferred matrix model on all sites in the dataset:
```
mpathic evaluate_model -m matrix_model.txt -i dataset.txt -o dataset_with_values.txt
```

Scan the *Escherichia coli* genome ([genome_ecoli.fa](examples/genome_ecoli.fa)) using the inferred matrix model:
```
mpathic scan_model -n 100 -m matrix_model.txt -i genome_ecoli.fa -o genome_sites.txt
```

Compute the predictive information of the inferred matrix model and the true model on the simulated data:
```
mpathic predictiveinfo -m matrix_model.txt -ds dataset.txt
mpathic predictiveinfo -m true_model.txt -ds dataset.txt
```

[documentation]: http://jbkinney.github.io/mpathic/
[preprint]: http://biorxiv.org/content/early/2016/05/21/054676


