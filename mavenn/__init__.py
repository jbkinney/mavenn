"""MAVE-NN software package."""
# The functions imported here are the ONLY "maven.xxx()" functions that
# users are expected to interact with

# Primary model class
from mavenn.src.model import Model

# For running functional tests
from mavenn.tests import run_tests

# Examples
from mavenn.src.examples import list_tutorials
from mavenn.src.examples import run_demo
from mavenn.src.examples import load_example_dataset
from mavenn.src.examples import load_example_model

# For loading models
from mavenn.src.utils import load

# For reproducible model inference
from mavenn.src.utils import set_seed

# For estimating the intrinsic information in a dataset
from mavenn.src.entropy import I_intrinsic

# For visualizing G-P maps
from mavenn.src.visualization import heatmap
from mavenn.src.visualization import heatmap_pairwise
