# The functions imported here are the ONLY "maven.xxx()" functions that
# users are expected to interact with

# For running functional tests
from mavenn.tests import run_tests

# Example demonstrations
from mavenn.src.examples import demo

# Example datasets
from mavenn.src.examples import example_dataset
# TODO: write example_model()

# Primary model class
from mavenn.src.model import Model

# For loading models
from mavenn.src.utils import load

# For estimating the intrinsic information in a dataset
from mavenn.src.entropy import estimate_instrinsic_information

# For visualizing G-P maps
from mavenn.src.visualization import heatmap
from mavenn.src.visualization import pairwise_heatmap
#from mavenn.src.landscape import get_1pt_variants



