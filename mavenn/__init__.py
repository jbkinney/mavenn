# The functions imported here are the ONLY "maven.xxx()" functions that
# users are expected to interact with

# For running functional tests
from mavenn.tests import run_tests

# Examples
from mavenn.src.examples import run_demo
from mavenn.src.examples import load_example_dataset
from mavenn.src.examples import load_example_model
# TODO: write load_example() to unify interface to examples.

# Primary model class
from mavenn.src.model import Model

# For loading models
from mavenn.src.utils import load

# For estimating the intrinsic information in a dataset
from mavenn.src.entropy import estimate_instrinsic_information

# For visualizing G-P maps
from mavenn.src.visualization import heatmap
from mavenn.src.visualization import pairwise_heatmap

# For generating lists of variants
from mavenn.src.landscape import get_1pt_variants
#TODO: write get_2p_variants()



