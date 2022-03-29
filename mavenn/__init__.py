"""MAVE-NN software package."""
# The functions imported here are the ONLY "maven.xxx()" functions that
# users are expected to interact with

# To regularize log calculations
import numpy as np
TINY = np.sqrt(np.finfo(np.float32).tiny)

# Primary model class
from mavenn.src.model import Model
from mavenn.src.model2 import Model as Model2

# Import GP layers class. This will allow users to create GP-map objects
# will can be passed into Model2.
from mavenn.src.layers import gpmap

# Import measurement process layers, which will allow users to instantiate
# them as MP objects that can then be passed to Model2.
from mavenn.src.layers import measurement_process_layers

# For running functional tests
from mavenn.tests import run_tests

# Examples
from mavenn.src.examples import list_tutorials
from mavenn.src.examples import run_demo
from mavenn.src.examples import load_example_dataset
from mavenn.src.examples import load_example_model

# For loading models
from mavenn.src.utils import load
from mavenn.src.utils import split_dataset

# For visualizing G-P maps
from mavenn.src.visualization import heatmap
from mavenn.src.visualization import heatmap_pairwise

